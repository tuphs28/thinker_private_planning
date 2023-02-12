import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import math

class ConvAttnLSTMCell(nn.Module):

    def __init__(self, input_dims, embed_dim, kernel_size=3, num_heads=8, mem_n=8, attn=True, attn_mask_b=3):

        super(ConvAttnLSTMCell, self).__init__()
        c, h, w = input_dims
        
        self.input_dims = input_dims
        self.linear = (h == w == 1)
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        if self.linear:
            self.main = nn.Linear(c + self.embed_dim, 5 * self.embed_dim)
        else:            
            self.main = nn.Conv2d(in_channels= c + self.embed_dim,
                                  out_channels= 5 * self.embed_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,)
        
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mem_n = mem_n
        self.head_dim = embed_dim // num_heads
        self.attn = attn
        self.attn_mask_b = attn_mask_b
        
        if self.attn:    
            if self.linear:
                self.proj = nn.Linear(c, self.embed_dim*3) 
                self.out = nn.Linear(self.embed_dim, self.embed_dim) 
            else:                
                self.proj = torch.nn.Conv2d(in_channels=c, out_channels=self.embed_dim*3, kernel_size=kernel_size, padding='same') 
                self.out = torch.nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=kernel_size, padding='same') 
                
            self.norm = nn.modules.normalization.LayerNorm((embed_dim, h, w), eps=1e-5)        
            self.pos_w = torch.nn.Parameter(torch.zeros(self.mem_n, h*w*embed_dim))
            self.pos_b = torch.nn.Parameter(torch.zeros(self.mem_n, self.num_heads))        
            torch.nn.init.xavier_uniform_(self.pos_w)
            torch.nn.init.uniform_(self.pos_b, -0.1, 0.1)            

    def forward(self, input, h_cur, c_cur, concat_k, concat_v, attn_mask):
        """
        Args:
          input (tensor): network input; shape (B, C, H, W)          
          h_cur (tensor): previous output; shape (B, embed_dim, H, W)
          c_cur (tensor): previous lstm state; shape (B, embed_dim, H, W)
          concat_k (tensor): previous attn k; shape (B, num_head, mem_n, total_dim)
          concat_v (tensor): previous attn v; shape (B, num_head, mem_n, total_dim)
          attn_mask (tensor): attn mask; shape (B * num_head, 1, mem_n)
        Return:
          h_next (tensor): current output; shape (B, embed_dim, H, W)
          c_next (tensor): current lstm state; shape (B, embed_dim, H, W)
          concat_k (tensor): current attn k; shape (B, num_head, mem_n, total_dim)
          concat_v (tensor): current attn v; shape (B, num_head, mem_n, total_dim)            
        """
        
        B = input.shape[0]
        combined = torch.cat([input, h_cur], dim=1)  # concatenate along channel axis
        if self.linear:
            combined_conv = self.main(combined[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        else:
            combined_conv = self.main(combined)           
        cc_i, cc_f, cc_o, cc_g, cc_a = torch.split(combined_conv, self.embed_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)        
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g 
        
        if self.attn:
            a = torch.sigmoid(cc_a)
            attn_out, concat_k, concat_v = self.attn_output(input, attn_mask, concat_k, concat_v)
            c_next = c_next + a * torch.tanh(attn_out)
            self.a = a
        else:
            concat_k, concat_v = None, None

        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next, concat_k, concat_v
    
    def attn_output(self, input, attn_mask, concat_k, concat_v):        
        b, c, h, w = input.shape
        tot_head_dim = h * w * self.embed_dim // self.num_heads
        
        if self.linear:
            kqv = self.proj(input[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        else:
            kqv = self.proj(input)            
        
        kqv_reshape = kqv.view(b * self.num_heads, self.head_dim * 3, h * w)
        k, q, v = torch.split(kqv_reshape, self.head_dim, dim=1) 
        k, q, v = [torch.flatten(x.unsqueeze(0), start_dim=2).transpose(0, 1) for x in [k, q, v]]  

        q_scaled = q / math.sqrt(q.shape[2])        
        k_pre = concat_k.view(b * self.num_heads, -1, tot_head_dim)        
        k = torch.cat([k_pre[:, 1:], k], axis=1)

        pos_w = (self.pos_w.unsqueeze(1).broadcast_to(self.mem_n, b, -1).contiguous().view(
            self.mem_n, b * self.num_heads, -1).transpose(0, 1))
        pos_b = (self.pos_b.unsqueeze(1).broadcast_to(self.mem_n, b, -1).contiguous().view(
            self.mem_n, b * self.num_heads).transpose(0, 1))

        k = k + pos_w

        v_pre = concat_v.view(b * self.num_heads, -1, tot_head_dim)
        v = torch.cat([v_pre[:, 1:], v], axis=1)
        
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask            
        attn_mask[:, :, -1] = self.attn_mask_b
        self.attn_mask = attn_mask
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))         
        attn_output_weights = attn_output_weights + pos_b.unsqueeze(1)        
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)                  
        self.attn_output_weights = attn_output_weights
        
        attn_output = torch.bmm(attn_output_weights, v)              
        attn_output = attn_output.transpose(0,1).view(b, self.embed_dim, h, w) 
        
        if self.linear:
            out = self.out(attn_output[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        else:
            out = self.out(attn_output)       
        out = out + input[:, :self.embed_dim]
        out = self.norm(out)
            
        ret_k = k.view(b, self.num_heads, self.mem_n, tot_head_dim)
        ret_v = v.view(b, self.num_heads, self.mem_n, tot_head_dim)
        
        return out, ret_k, ret_v

class ConvAttnLSTM(nn.Module):

    def __init__(self, h, w, input_dim, hidden_dim, kernel_size, 
            num_layers, num_heads, mem_n, attn, attn_mask_b, grad_scale):
    
        super(ConvAttnLSTM, self).__init__()
        
        self.h = h
        self.w = w
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mem_n = mem_n
        self.grad_scale = grad_scale
        self.attn = attn
        self.tot_head_dim = h * w * hidden_dim // num_heads        

        layers = []
        
        for i in range(0, self.num_layers):
            layers.append(ConvAttnLSTMCell(input_dims=(input_dim+hidden_dim, self.h, self.w),
                                           embed_dim=self.hidden_dim,
                                           kernel_size=self.kernel_size,
                                           num_heads=num_heads,
                                           mem_n=mem_n,
                                           attn=attn,
                                           attn_mask_b=attn_mask_b,))

        self.layers = nn.ModuleList(layers)
    
    def init_state(self, bsz, device=None):        
        core_state = (torch.zeros(self.num_layers, bsz, self.hidden_dim, self.h, self.w, device=device),
                      torch.zeros(self.num_layers, bsz, self.hidden_dim, self.h, self.w, device=device),)
        if self.attn:
            core_state = core_state + (torch.zeros(self.num_layers, bsz, self.num_heads, self.mem_n, self.tot_head_dim, device=device),
                      torch.zeros(self.num_layers, bsz,  self.num_heads, self.mem_n, self.tot_head_dim, device=device),
                      torch.ones(1, bsz, self.mem_n, device=device).bool())
        return core_state
    
    def forward(self, x, core_state, notdone, notdone_attn=None):        
        b, c, h, w = x.shape        
        out = core_state[0][-1] * notdone.float().view(b, 1, 1, 1)  
        
        if notdone_attn is None: notdone_attn = notdone        
        if self.attn:  
            src_mask = core_state[4][0]
            src_mask[~(notdone_attn.bool()), :] = True
            src_mask[:, :-1] = src_mask[:, 1:].clone().detach()
            src_mask[:, -1] = False                        
            new_src_mask = src_mask.unsqueeze(0)
            src_mask_reshape = src_mask.view(b, 1, 1, -1).broadcast_to(b, self.num_heads, 1, -1).contiguous().view(b * self.num_heads, 1, -1)               
        else:
            src_mask_reshape = None
                
        core_out = []
        new_core_state = ([], [], [], []) if self.attn else ([], [])
        for n, cell in enumerate(self.layers):
            cell_input = torch.concat([x, out], dim=1)
            h_cur = core_state[0][n] * notdone.float().view(b, 1, 1, 1)   
            c_cur = core_state[1][n] * notdone.float().view(b, 1, 1, 1)    
            concat_k_cur = core_state[2][n] if self.attn else None
            concat_v_cur = core_state[3][n] if self.attn else None

            h_next, c_next, concat_k, concat_v = cell(cell_input, h_cur, c_cur, 
                                                      concat_k_cur, concat_v_cur, src_mask_reshape)
            if self.grad_scale < 1 and h_next.requires_grad:
                h_next.register_hook(lambda grad: grad * self.grad_scale)
                c_next.register_hook(lambda grad: grad * self.grad_scale)
            if self.grad_scale < 1 and self.attn and concat_k.requires_grad:    
                concat_k.register_hook(lambda grad: grad * self.grad_scale)
                concat_v.register_hook(lambda grad: grad * self.grad_scale)

            new_core_state[0].append(h_next)
            new_core_state[1].append(c_next)     
            if self.attn:
                new_core_state[2].append(concat_k)                
                new_core_state[3].append(concat_v)
            out = h_next
        
        core_state = tuple(torch.cat([u.unsqueeze(0) for u in v]) for v in new_core_state)
        if self.attn:
            core_state = core_state + (new_src_mask,)
        
        core_out = out.unsqueeze(0)
        return core_out, core_state
