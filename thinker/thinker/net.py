from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from thinker import util
from thinker.core.rnn import ConvAttnLSTM
from math import prod

ActorOut = namedtuple('ActorOut', ['policy_logits', 'im_policy_logits', 'reset_policy_logits', 
    'action', 'im_action', 'reset_action', 'baseline', 'baseline_enc_s', 'reg_loss'])

OutNetOut = namedtuple('OutNetOut', ['single_rs',  'rs', 'r_enc_logits', 'dones',
                                     'done_logits', 'vs', 'v_enc_logits', 'logits', 'r_state'])
ModelNetOut = namedtuple('ModelNetOut', ['single_rs',  'rs', 'r_enc_logits', 'pred_xs',
                                         'dones', 'done_logits', 'vs', 'v_enc_logits', 
                                         'logits', 'hs', 'pred_zs', 'pred_z_logits', 
                                         'true_zs', 'true_z_logits', 'r_state', ])

def add_hw(x, h, w):
    return x.unsqueeze(-1).unsqueeze(-1).broadcast_to(x.shape + (h,w))

def mlp(input_size, layer_sizes, output_size, output_activation=nn.Identity, 
        activation=nn.ReLU, momentum=0.1, zero_init=False, norm=True):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if norm: layers.append(nn.BatchNorm1d(sizes[i + 1], momentum=momentum))
            layers.append(act())
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]
    if zero_init:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)
    return nn.Sequential(*layers)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=False, dilation=dilation,)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, outplanes=None, stride=1, downsample=None, disable_bn=False):
        super().__init__()
        if outplanes is None: outplanes = inplanes 
        if disable_bn:
            norm_layer = nn.Identity
        else:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, outplanes)
        self.bn2 = norm_layer(outplanes)
        self.skip_conv = (outplanes != inplanes)
        self.stride = stride
        if outplanes != inplanes:
            if downsample is None:
                self.conv3 = conv1x1(inplanes, outplanes)
            else:
                self.conv3 = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip_conv:
            out += self.conv3(identity)
        else:
            out += identity
        out = self.relu(out)
        return out


class ActorEncoder(nn.Module):
    def __init__(self, input_shape, num_actions, frame_encode, double_encode, compress_size, drc, rnn_grad_scale):
        super(ActorEncoder, self).__init__()
        self.input_shape = input_shape        
        self.frame_encode = frame_encode
        self.double_encode = double_encode
        self.compress = compress_size > 0
        self.compress_size = compress_size        
        self.drc = drc
        self.rnn_grad_scale = rnn_grad_scale        
        encoder_out_channels = 32

        if frame_encode:
            down_scale_c = 4
            self.frame_encoder = FrameEncoder(input_shape=input_shape, num_actions=num_actions, 
                    down_scale_c=down_scale_c, concat_action=False)
            input_shape = self.frame_encoder.out_shape

        self.three_d_input = len(self.input_shape) == 3            
        if not self.three_d_input:
            # the input is not 3D, we use MLP layers
            in_size = prod(self.input_shape)
            self.core = nn.Sequential(nn.Linear(in_size, 256), nn.ReLU(),
                                      nn.Linear(256, 256), nn.ReLU())
            core_out_size = 256
        else:
            in_channels = input_shape[0]
            in_hw = input_shape[1]

            # input shape here is (in_channels, in_hw, in_hw)
            if not drc:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=encoder_out_channels*2, kernel_size=3, padding='same'), 
                    nn.ReLU(), 
                    nn.Conv2d(in_channels=encoder_out_channels*2, out_channels=encoder_out_channels, kernel_size=3, padding='same'), 
                    nn.ReLU())            
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=encoder_out_channels, kernel_size=3, padding='same'), 
                    nn.ReLU())
                self.drc_core = ConvAttnLSTM(h=in_hw, w=in_hw,
                                            input_dim=encoder_out_channels, hidden_dim=encoder_out_channels, kernel_size=3, num_layers=3, 
                                            num_heads=8, mem_n=0, attn=False, attn_mask_b=0.,
                                            grad_scale=self.rnn_grad_scale)                   
            core_out_size = encoder_out_channels * in_hw * in_hw
            
        if self.compress:            
            self.fc = nn.Sequential(nn.Linear(core_out_size, compress_size), nn.ReLU())

        self.out_size = core_out_size if not self.compress else compress_size
        if self.double_encode: self.out_size *= 2

        self.initial_state(1)

    def initial_state(self, batch_size, device=None):
        if self.drc:
            state = self.drc_core.init_state(batch_size, device=device)
            self.core_state_sep_ind = len(state)
            if self.double_encode:
                state = state + self.actor_encoder.drc_core.init_state(batch_size, device=device)
        else:
            state = None
        return state
    
    def forward(self, x, mask=None, core_state=(), done=None):
        """encode the state or model's encoding inside the actor network
        args:
            x: input tensor of shape (T, B, C, H, W); can be state or model's encoding,
            mask: mask tensor of shape (T, B); boolean
            core_state: core_state for drc module (only when drc is enabled)
            done: done tensor of shape (T, B); boolean
        return:
            output: output tensor of shape (T*B, self.out_size)"""
        if not self.double_encode:
            return self.forward_single(x, mask, core_state, done)
        else:
            _, _, C, *_ = x.shape
            enc_1, core_state_1 = self.forward_single(x[:, :, :C//2], mask, core_state[:self.core_state_sep_ind] if self.drc else None, done)
            enc_2, core_state_2 = self.forward_single(x[:, :, C//2:], mask, core_state[self.core_state_sep_ind:] if self.drc else None, done)
            enc = torch.concat([enc_1, enc_2], dim=1)
            if self.drc:
                core_state = core_state_1 + core_state_2
            else:
                core_state = None
            return enc, core_state    
        
    def forward_single(self, x, mask=None, core_state=(), done=None):
        T, B, *_ = x.shape
        x = x.float()
        if mask is not None:
            x = x * mask.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = torch.flatten(x, 0, 1)  

        if self.frame_encode:
            x, _ = self.frame_encoder(x, actions=None)
        if not self.three_d_input:
            x = torch.flatten(x, start_dim=1)
            x = self.core(x)
            core_state = None
        else:
            x = self.conv(x)
            if self.drc:
                core_input = x.view(*((T, B) + x.shape[1:]))
                core_output_list = []
                notdone = ~(done.bool())
                for n, (input, nd) in enumerate(zip(core_input.unbind(), notdone.unbind())):
                    for t in range(3):                
                        if t > 0: nd = torch.ones(B, device=x.device, dtype=torch.bool)
                        nd = nd.view(-1)      
                        output, core_state = self.drc_core(input, core_state, nd, nd) # output shape: 1, B, core_output_size    
                    core_output_list.append(output)              
                core_output = torch.cat(core_output_list)   
                x = torch.flatten(core_output, start_dim=0, end_dim=1)
            else:
                core_state = None
            x = torch.flatten(x, start_dim=1)
        if self.compress: 
            x = self.fc(x)
        return x, core_state

class ActorNetBase(nn.Module):    
    def __init__(self, obs_shape, gym_obs_shape, num_actions, flags):

        super(ActorNetBase, self).__init__()
        self.obs_shape = obs_shape
        self.gym_obs_shape = gym_obs_shape
        self.num_actions = num_actions  
        
        self.tran_t = flags.tran_t                   # number of recurrence of RNN        
        self.tran_mem_n = flags.tran_mem_n           # size of memory for the attn modules
        self.tran_layer_n = flags.tran_layer_n       # number of layers
        self.tran_lstm_no_attn = flags.tran_lstm_no_attn  # to use attention in lstm or not
        self.attn_mask_b = flags.tran_attn_b         # atention bias for current position
        self.tran_dim = flags.tran_dim               # size of transformer / LSTM embedding dim        
        self.num_rewards = 2 if (flags.reward_type == 1) else 1 # dim of rewards (1 for vanilla; 2 for planning rewards)
        self.actor_see_p = flags.actor_see_p         # probability of allowing actor to see state
        self.actor_see_encode = flags.actor_see_encode # Whether the actor see the model encoded state or the raw env state
        self.actor_see_double_encode = flags.actor_see_double_encode # Whether the actor see the model encoded state or the raw env state
        self.actor_see_h = flags.actor_see_h # Whether the actor see the hidden state also
        self.actor_encode_concat_type = flags.actor_encode_concat_type # Type of concating the encoding to model's output
        self.actor_drc = flags.actor_drc             # Whether to use drc in encoding state
        self.rnn_grad_scale = flags.rnn_grad_scale   # Grad scale for hidden state in RNN
        self.reward_transform = flags.reward_transform # Whether to use reward transform as in MuZero
        self.model_type_nn = flags.model_type_nn
        self.model_size_nn = flags.model_size_nn     

        # encoder for state or encoding output        
        if self.actor_see_p > 0:
            if not self.actor_see_encode:
                input_shape = gym_obs_shape
            else:
                if self.model_type_nn in [0, 1,  2,  3]:
                    in_channels= 128 if self.model_type_nn in [0, 1, 3] else 64
                    input_shape = (in_channels, gym_obs_shape[1]//16, gym_obs_shape[2]//16)
                elif self.model_type_nn == 4: 
                    input_shape = 32 * flags.model_size_nn * 32 * flags.model_size_nn
                    if self.actor_see_h:
                        input_shape = input_shape + 512
                    input_shape = (input_shape,)
            compress_size = 128 if self.actor_encode_concat_type == 1 else 256
            self.actor_encoder = ActorEncoder(input_shape=input_shape, num_actions=num_actions,
                frame_encode=not self.actor_see_encode, double_encode=self.actor_see_double_encode, 
                compress_size=compress_size, drc=self.actor_drc, rnn_grad_scale=self.rnn_grad_scale)        

        in_channels = self.obs_shape[0]
        if self.actor_see_p > 0 and self.actor_encode_concat_type == 1:
            in_channels = in_channels + self.actor_encoder.out_size

        self.initial_enc = nn.Sequential(nn.Linear(in_channels, self.tran_dim), 
                                              nn.ReLU(),
                                              nn.Linear(self.tran_dim, self.tran_dim), 
                                              nn.ReLU())

        self.core = ConvAttnLSTM(h=1, w=1,
                                input_dim=self.tran_dim, hidden_dim=self.tran_dim,
                                kernel_size=1, num_layers=self.tran_layer_n,
                                num_heads=8, mem_n=self.tran_mem_n, attn=not self.tran_lstm_no_attn,
                                attn_mask_b=self.attn_mask_b, grad_scale=self.rnn_grad_scale)                         
        
        last_out_size = 256           
        self.fc = nn.Sequential(nn.Linear(self.tran_dim, last_out_size),  nn.ReLU())              
        if self.actor_see_p > 0 and self.actor_encode_concat_type == 0:             
            last_out_size = last_out_size + self.actor_encoder.out_size
            
        self.im_policy = nn.Linear(last_out_size, self.num_actions)        
        self.policy = nn.Linear(last_out_size, self.num_actions)       
        self.reset = nn.Linear(last_out_size, 2)    
        self.baseline = nn.Linear(last_out_size, self.num_rewards)        

        if self.reward_transform:
            self.reward_tran = RewardTran(vec=False)
        self.initial_state(1) # just for setting core_state_sep_ind

    def initial_state(self, batch_size, device=None):
        state = self.core.init_state(batch_size, device=device) 
        self.core_state_sep_ind = len(state)
        if self.actor_see_p > 0 and self.actor_drc:
            state = state + self.actor_encoder.initial_state(batch_size, device=device)
        return state

    def forward(self, obs, core_state=()):
        """one-step forward for the actor;
        args:
            obs (EnvOut):
                model_out (tensor): model output with shape (T x B x C) or (B x C)
                done  (tensor): if episode ends with shape (T x B) or (B)
                cur_t (tensor): current planning step with shape (T x B) 
                and other environment output that is not used.
        return:
            ActorOut:
                policy_logits (tensor): logits of real action (T x B x |A|)
                im_policy_logits (tensor): logits of imagine action (T x B x |A|)
                reset_policy_logits (tensor): logits of real action (T x B x 2)
                action (tensor): sampled real action (non-one-hot form) (T x B)
                im_action (tensor): sampled imagine action (non-one-hot form) (T x B)
                reset_action (tensor): sampled reset action (non-one-hot form) (T x B)
                baseline (tensor): prediced baseline (T x B x 1) or (T x B x 2)
                reg_loss (tensor): regularization loss (T x B)
        """
                
        x = obs.model_out
        done = obs.done
        
        if len(x.shape) == 2: x = x.unsqueeze(0)
        if len(done.shape) == 1: done = done.unsqueeze(0)  
            
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.  

        if self.actor_see_p > 0:
            encoder_out, core_state_ = self.actor_encoder(obs.model_encodes if self.actor_see_encode else obs.gym_env_out, 
                                             mask=obs.see_mask if self.actor_see_p < 1 else None, 
                                             core_state = core_state[self.core_state_sep_ind:],
                                             done=done)
            
        if self.actor_encode_concat_type == 1 and self.actor_see_p > 0:
            x = torch.concat([x, encoder_out], dim=1)
        x = self.initial_enc(x)                

        core_input = x.view(*((T, B) + x.shape[1:]))
        core_output_list = []
        core_state = core_state[:self.core_state_sep_ind]
        notdone = ~(done.bool())
        core_input = core_input.unsqueeze(-1).unsqueeze(-1)
        for n, (input, nd) in enumerate(zip(core_input.unbind(), notdone.unbind())):  
            for t in range(self.tran_t):                
                if t > 0: nd = torch.ones(B, device=x.device, dtype=torch.bool)                 
                nd = nd.view(-1)      
                output, core_state = self.core(input, core_state, nd, nd) # output shape: 1, B, core_output_size                 
            core_output_list.append(output)                             
        core_output = torch.cat(core_output_list)              
        core_output = torch.flatten(core_output, 0, 1)        
        core_output = torch.flatten(core_output, start_dim=1)
        x = self.fc(core_output)

        if self.actor_encode_concat_type == 0 and self.actor_see_p > 0:
            x = torch.concat([x, encoder_out], dim=1)
   
        policy_logits = self.policy(x)
        im_policy_logits = self.im_policy(x)
        reset_policy_logits = self.reset(x)

        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        im_action = torch.multinomial(F.softmax(im_policy_logits, dim=1), num_samples=1)
        reset_action = torch.multinomial(F.softmax(reset_policy_logits, dim=1), num_samples=1)
        
        if not self.reward_transform:
            baseline = self.baseline(x)
        else:            
            baseline_enc_s = self.baseline(x)
            baseline = self.reward_tran.decode(baseline_enc_s)
                   
        reg_loss = (1e-3 * torch.sum(policy_logits**2, dim=-1) / 2 + 
                    1e-3 * torch.sum(im_policy_logits**2, dim=-1) / 2 + 
                    1e-3 * torch.sum(reset_policy_logits**2, dim=-1) / 2 + 
                    1e-5 * torch.sum(core_output**2, dim=-1) / 2)
        reg_loss = reg_loss.view(T, B)
        
        policy_logits = policy_logits.view(T, B, self.num_actions)
        im_policy_logits = im_policy_logits.view(T, B, self.num_actions)
        reset_policy_logits = reset_policy_logits.view(T, B, 2)
        
        action = action.view(T, B)      
        im_action = im_action.view(T, B)      
        reset_action = reset_action.view(T, B)                 

        baseline_enc_s = baseline_enc_s.view(T, B, self.num_rewards) if self.reward_transform else None
        baseline = baseline.view(T, B, self.num_rewards)        

        actor_out = ActorOut(policy_logits=policy_logits,                         
                             im_policy_logits=im_policy_logits,                         
                             reset_policy_logits=reset_policy_logits,     
                             action=action,     
                             im_action=im_action,
                             reset_action=reset_action,
                             baseline_enc_s=baseline_enc_s,
                             baseline=baseline, 
                             reg_loss=reg_loss,)       
        
        if self.actor_see_p > 0 and self.actor_drc:
            core_state = core_state + core_state_
        return actor_out, core_state

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

def ActorNet(obs_shape, gym_obs_shape, num_actions, flags):
    if flags.actor_net_ver == 1:
        return ActorNetBase(obs_shape, gym_obs_shape, num_actions, flags)
    elif flags.actor_net_ver == 0:
        from thinker.legacy import LegacyActorNet
        return LegacyActorNet(obs_shape, gym_obs_shape, num_actions, flags)

# Model Network

class FrameEncoder(nn.Module):    
    def __init__(self, num_actions, input_shape, type_nn=0, 
                 size_nn=1, down_scale_c=2, concat_action=True, decoder=False):
        super(FrameEncoder, self).__init__() 
        self.num_actions = num_actions
        self.type_nn = type_nn
        self.size_nn = size_nn
        self.down_scale_c = down_scale_c
        self.decoder = decoder
        frame_channels, h, w = input_shape

        if type_nn in [0, 1]:
            self.concat_action = concat_action
        elif type_nn in [2, 3, 4]:
            self.concat_action = False        

        if self.concat_action:
            in_channels=frame_channels+num_actions
        else:
            in_channels=frame_channels        

        if type_nn in [0, 1]:
            # mu zero
            if type_nn == 0:
                n_block = 1  * self.size_nn
            elif type_nn == 1:
                n_block = 2 * self.size_nn

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128//down_scale_c, kernel_size=3, stride=2, padding=1) 
            res = [ResBlock(inplanes=128//down_scale_c) for _ in range(n_block)] # Deep: 2 blocks here
            self.res1 = nn.Sequential(*res)
            self.conv2 = nn.Conv2d(in_channels=128//down_scale_c, out_channels=256//down_scale_c, 
                                kernel_size=3, stride=2, padding=1) 
            res =  [ResBlock(inplanes=256//down_scale_c) for _ in range(n_block)] # Deep: 3 blocks here
            self.res2 = nn.Sequential(*res)
            self.avg1 = nn.AvgPool2d(3, stride=2, padding=1)
            res = [ResBlock(inplanes=256//down_scale_c) for _ in range(n_block)] # Deep: 3 blocks here
            self.res3 = nn.Sequential(*res)
            self.avg2 = nn.AvgPool2d(3, stride=2, padding=1)
            self.out_shape = (256//down_scale_c, h//16, w//16)

            if decoder:
                d_conv = [ResBlock(inplanes=256//down_scale_c) for _ in range(n_block)]
                kernel_sizes = [4, 4, 4, 4]
                conv_channels = [frame_channels, 128//down_scale_c, 256//down_scale_c, 256//down_scale_c, 256//down_scale_c]
                for i in range(4):
                    if i in [1, 3]:
                        d_conv.extend([ResBlock(inplanes=conv_channels[4-i]) for _ in range(n_block)])
                    d_conv.append(nn.ReLU())
                    d_conv.append(nn.ConvTranspose2d(conv_channels[4-i], conv_channels[4-i-1], 
                                          kernel_size=kernel_sizes[i], stride=2, padding=1))    
                self.d_conv = nn.Sequential(*d_conv)

        elif type_nn in [2, 3]:
            # efficient zero
            num_block = 1 if type_nn in [2] else (2 * self.size_nn)
            out_channels = 64 if type_nn in [2] else 128
            self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels // 2)
            self.resblocks1 = nn.ModuleList(
                [ResBlock(out_channels // 2, out_channels // 2) for _ in range(num_block)]
            )
            self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False,)
            self.downsample_block = ResBlock(out_channels // 2, out_channels, stride=2, downsample=self.conv2)
            self.resblocks2 = nn.ModuleList(
                [ResBlock(out_channels, out_channels) for _ in range(num_block)]
            )
            self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.resblocks3 = nn.ModuleList(
                [ResBlock(out_channels, out_channels) for _ in range(num_block)]
            )
            self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)   
            self.resblocks4 = nn.ModuleList(
                [ResBlock(out_channels, out_channels) for _ in range(num_block)]
            )              
            self.out_shape = (out_channels, h//16, w//16)
        elif type_nn in [4]:
            # dreamer
            n_block = 1
            l = [nn.Conv2d(in_channels=in_channels, out_channels=128//down_scale_c, kernel_size=3, stride=2, padding=1)]
            l.extend([ResBlock(inplanes=128//down_scale_c) for _ in range(n_block)])
            l.append(nn.Conv2d(in_channels=128//down_scale_c, out_channels=256//down_scale_c, 
                                kernel_size=3, stride=2, padding=1))
            l.extend([ResBlock(inplanes=256//down_scale_c) for _ in range(n_block)])
            l.append(nn.AvgPool2d(3, stride=2, padding=1))
            l.extend([ResBlock(inplanes=256//down_scale_c) for _ in range(n_block)])
            l.append(nn.AvgPool2d(3, stride=2, padding=1))
            self.conv = nn.Sequential(*l)
            self.conv_out_shape = (256//down_scale_c, h//16, w//16)

            s_size = 32 * self.size_nn
            mlp_in = prod(self.conv_out_shape)
            mlp = []
            for i in range(1):                
                mlp.append(nn.Linear(mlp_in + 512 if i == 0 else 400, 400))
                mlp.append(nn.ELU())
            mlp.append(nn.Linear(400, s_size*s_size))
            self.mlp = nn.Sequential(*mlp)
            self.out_shape = (s_size, s_size)

            if decoder:
                d_mlp = [nn.Linear(s_size*s_size + 512, 400)]
                for i in range(1):
                    d_mlp.append(nn.ELU())
                    d_mlp.append(nn.Linear(400, 400 if i < 0 else mlp_in))
                self.d_mlp = nn.Sequential(*d_mlp)
                    
                d_conv = [ResBlock(inplanes=256//down_scale_c) for _ in range(n_block)]
                kernel_sizes = [4, 4, 4, 4]
                conv_channels = [frame_channels, 128//down_scale_c, 256//down_scale_c, 256//down_scale_c, 256//down_scale_c]
                for i in range(4):
                    if i in [1, 3]:
                        d_conv.extend([ResBlock(inplanes=conv_channels[4-i]) for _ in range(n_block)])
                    d_conv.append(nn.ReLU())
                    d_conv.append(nn.ConvTranspose2d(conv_channels[4-i], conv_channels[4-i-1], 
                                          kernel_size=kernel_sizes[i], stride=2, padding=1))    
                self.d_conv = nn.Sequential(*d_conv)
    
    def forward(self, x, actions, h=None, rescale=True, flatten=False):      
        """
        Args:
          x (tensor): frame with shape B, C, H, W        
          action (tensor): action with shape B, num_actions (in one-hot)
        """
        if flatten:
            input_shape = x.shape
            x = x.view((x.shape[0]*x.shape[1],) + x.shape[2:])
            actions = actions.view((actions.shape[0]*actions.shape[1],) + actions.shape[2:])

        b, _, _, _ = x.shape
        if rescale:
            x = x.float() / 255.0    
        if self.type_nn in [0, 1]:
            if self.concat_action:
                actions = actions.unsqueeze(-1).unsqueeze(-1).tile([1, 1, x.shape[2], x.shape[3]])        
                x = torch.concat([x, actions], dim=1)
            x = F.relu(self.conv1(x))
            x = self.res1(x)
            x = F.relu(self.conv2(x))
            x = self.res2(x)
            x = self.avg1(x)
            x = self.res3(x)
            x = self.avg2(x)
            z, z_logit = x, None
        elif self.type_nn in [2, 3]:
            x = self.conv1(x)
            x = self.bn1(x)
            x = nn.functional.relu(x)
            for block in self.resblocks1:
                x = block(x)
            x = self.downsample_block(x)
            for block in self.resblocks2:
                x = block(x)
            x = self.pooling1(x)
            for block in self.resblocks3:
                x = block(x)
            x = self.pooling2(x)
            for block in self.resblocks4:
                x = block(x)
            z, z_logit = x, None            
        elif self.type_nn == 4:    
            s_size = 32 * self.size_nn
            x = self.conv(x)
            x = torch.flatten(x, start_dim=1)
            x = torch.concat([x, h], dim=1)
            z_logit = self.mlp(x)
            z_logit = z_logit.view(b, s_size, s_size)
            z_p = F.softmax(z_logit, dim=2)
            z = torch.multinomial(z_p.view(b*s_size, s_size), num_samples=1).view(b, s_size)
            z = F.one_hot(z, num_classes=s_size)
            z = z + z_p - z_p.detach()
            #z = z_p
        if flatten:
            z = z.view(input_shape[:2]+z.shape[1:])
            if z_logit is not None:
                z_logit = z_logit.view(input_shape[:2]+z_logit.shape[1:])
        return z, z_logit
    
    def decode(self, z, h=None, flatten=False):
        """
        Args:
          z (tensor): encoding with shape B, *
        """
        if flatten:
            input_shape = z.shape
            z = z.view((z.shape[0]*z.shape[1],) + z.shape[2:])
            if h is not None: h = h.view((h.shape[0]*h.shape[1],) + h.shape[2:])
        b = z.shape[0]
        if self.type_nn in [0, 1]:
            x = self.d_conv(z)                        

        elif self.type_nn in [4]:            
            z = z.view(b, -1)
            x = torch.concat([z, h], dim=1)
            x = self.d_mlp(x)
            x = x.reshape((b,) + self.conv_out_shape)
            x = self.d_conv(x)                        

        else: raise Exception("%d model_type_nn does not support decoder" % self.type_nn)

        if flatten:
            x = x.view(input_shape[:2]+x.shape[1:])
        return x
    
class DynamicModel(nn.Module):
    def __init__(self, num_actions, inplanes, type_nn=0, size_nn=1, 
                 outplanes=None, disable_half_grad=False, disable_bn=False):        
        super(DynamicModel, self).__init__()
        self.num_actions = num_actions     
        self.inplanes = inplanes
        self.type_nn = type_nn
        self.size_nn = size_nn        
        self.disable_half_grad = disable_half_grad
        if outplanes is None: outplanes = inplanes

        if type_nn == 0:
            res = [ResBlock(inplanes=inplanes+num_actions, 
                            outplanes=outplanes, disable_bn=disable_bn)] + [
                    ResBlock(inplanes=outplanes, disable_bn=disable_bn) for i in range(4 * self.size_nn)]
            self.res = nn.Sequential(*res)
            self.outplanes = outplanes 
        elif type_nn == 1:             
            res = [ResBlock(inplanes=inplanes if i == 0 else outplanes, 
                            outplanes=outplanes,
                            disable_bn=disable_bn) for i in range(15)] + [
                    ResBlock(inplanes=outplanes, 
                             outplanes=outplanes*num_actions,
                             disable_bn=disable_bn)]
            self.res = nn.Sequential(*res)    
            self.outplanes = outplanes 
        elif type_nn in [2, 3]:
            num_block = 1 if type_nn == 2 else (4 * self.size_nn)
            self.conv1 =  conv3x3(inplanes+num_actions, outplanes)
            self.bn1 = nn.BatchNorm2d(outplanes) if not disable_bn else nn.Identity()
            res = [ResBlock(inplanes=outplanes, disable_bn=disable_bn) for i in range(num_block)]
            self.res = nn.Sequential(*res)   
            self.outplanes = outplanes 
        elif type_nn in [4]:
            rnn_size = 512
            self.mlp = nn.Sequential(nn.Linear(inplanes+num_actions, rnn_size))
            self.rnn = nn.GRUCell(rnn_size, rnn_size)
            self.outplanes = rnn_size
    
    def forward(self, z, h, actions): 
        x = h
        if self.type_nn in [0, 1, 2, 3]:
            b, c, height, width = x.shape      
        elif self.type_nn in [4]:
            b, c = h.shape      
        if self.training and self.type_nn in [0, 1, 2, 3] and not self.disable_half_grad:
            # no half-gradient for dreamer net
            x.register_hook(lambda grad: grad * 0.5)
        if self.type_nn == 0:
            actions = actions.unsqueeze(-1).unsqueeze(-1).tile  ([1, 1, x.shape[2], x.shape[3]])        
            x = torch.concat([x, actions], dim=1)
            out = self.res(x)
        elif self.type_nn == 1:            
            res_out = self.res(x).view(b, self.num_actions, self.outplanes, height, width)        
            actions = actions.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = torch.sum(actions * res_out, dim=1)
        elif self.type_nn in [2,3]:  
            actions = actions.unsqueeze(-1).unsqueeze(-1).tile([1, 1, x.shape[2], x.shape[3]])        
            out = torch.concat([x, actions], dim=1)
            out = self.conv1(out)
            out = self.bn1(out)
            out = out + x
            out = nn.functional.relu(out)
            out = self.res(out)
        elif self.type_nn in [4]:
            y = z.view(b, -1)
            y = torch.concat([y, actions], dim=1)
            y = self.mlp(y)
            out = self.rnn(y, x)
        return out
    
    def init_state(self, bsz, device=None):
        if self.type_nn in [4]:
            return (torch.zeros(bsz, self.outplanes, device=device),)
        else:
            return ()
    
class Output_rvpi(nn.Module):   
    def __init__(self, num_actions, input_shape, value_prefix, max_unroll_length, 
            reward_transform, stop_vpi_grad, zero_init, type_nn, size_nn,
            predict_v_pi=True, predict_r=True, predict_done=False, disable_bn=False):         
        super(Output_rvpi, self).__init__()    
        
        self.input_shape = input_shape
        self.type_nn = type_nn
        self.size_nn = size_nn
        self.value_prefix = value_prefix
        self.max_unroll_length = max_unroll_length
        self.reward_transform = reward_transform
        self.stop_vpi_grad = stop_vpi_grad
        self.predict_v_pi = predict_v_pi
        self.predict_r = predict_r
        self.predict_done = predict_done

        if self.type_nn in [0, 1, 2, 3]:
            c, h, w = input_shape
        elif self.type_nn in [4]:
            c, = input_shape

        if self.reward_transform:
            self.reward_tran = RewardTran(vec=True)
            out_n = self.reward_tran.encoded_n
        else:
            out_n = 1    

        layer_norm = nn.BatchNorm2d if not disable_bn else nn.Identity

        if self.type_nn in [0, 1]:
            self.conv1 = nn.Conv2d(in_channels=c, out_channels=c//2, kernel_size=3, padding='same') 
            self.conv2 = nn.Conv2d(in_channels=c//2, out_channels=c//4, kernel_size=3, padding='same') 
            fc_in = h * w * (c // 4)  

            if predict_v_pi:
                self.fc_logits = nn.Linear(fc_in, num_actions)         
                self.fc_v = nn.Linear(fc_in, out_n) 
                if zero_init:                
                    nn.init.constant_(self.fc_v.weight, 0.)
                    nn.init.constant_(self.fc_v.bias, 0.)
                    nn.init.constant_(self.fc_logits.weight, 0.)
                    nn.init.constant_(self.fc_logits.bias, 0.) 

            if predict_done:
                self.fc_done = nn.Linear(fc_in,1) 
                if zero_init:                
                    nn.init.constant_(self.fc_done.weight, 0.)

            if predict_r and not self.value_prefix:
                self.fc_r = nn.Linear(fc_in, out_n)      
                if zero_init:
                    nn.init.constant_(self.fc_r.weight, 0.)
                    nn.init.constant_(self.fc_r.bias, 0.)

        elif self.type_nn in [2, 3]:
            num_block = 1 if self.type_nn  == 2 else 2 * self.size_nn
            out_channels = 16 if self.type_nn  == 2 else 16 * self.size_nn
            self.resblocks = nn.ModuleList(
                [ResBlock(c, c) for _ in range(num_block)]
            )
            fc_in = h * w * out_channels            
            if predict_v_pi:
                self.conv1x1_v = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1)
                self.bn_v = layer_norm(out_channels) 
                self.fc_v = mlp(fc_in, [32], out_n, zero_init=zero_init, norm=not disable_bn)                   
                self.conv1x1_logits = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1)            
                self.bn_logits = layer_norm(out_channels) 
                self.fc_logits = mlp(fc_in, [32], num_actions, zero_init=zero_init, norm=not disable_bn)                 

            if predict_done:
                self.conv1x1_done = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1)
                self.bn_done = layer_norm(out_channels)
                self.fc_done = mlp(fc_in, [32], 1, zero_init=zero_init, norm=not disable_bn)            

            if predict_r and not self.value_prefix:
                self.conv1x1_r = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1)
                self.bn_r = layer_norm(out_channels) 
                self.fc_r = mlp(fc_in, [32], out_n, zero_init=zero_init, norm=not disable_bn)

        elif self.type_nn in [4]:
            assert predict_v_pi and predict_r
            fc_in = c
            self.fc_v = mlp(fc_in, [400, 400], out_n, activation=nn.ELU, zero_init=zero_init, norm=False)            
            self.fc_logits = mlp(fc_in, [400, 400], num_actions, activation=nn.ELU, zero_init=zero_init, norm=False)            
            if predict_done: self.fc_done = mlp(fc_in, [400, 400], 1, activation=nn.ELU, zero_init=zero_init, norm=False)            
            if not self.value_prefix:
                self.fc_r = mlp(fc_in, [400, 400], out_n, activation=nn.ELU, zero_init=zero_init, norm=False)            
            
        if predict_r and self.value_prefix:
            if self.type_nn in [0, 1, 2, 3]:
                self.conv1x1_reward = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=1)            
                self.bn_r_1 = layer_norm(16)
                self.lstm_input_size=16*input_shape[1]*input_shape[2]
            elif self.type_nn in [4]:
                self.lstm_input_size = c
            self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=512)
            self.bn_r_2 = nn.Identity() if disable_bn else nn.BatchNorm1d(512)
            out_n = self.reward_tran.encoded_n if self.reward_transform else 1
            self.fc_r = mlp(512, [64], out_n, zero_init=zero_init, norm=not disable_bn)                
        
    def forward(self, z, h, predict_reward=True, state=()):   
        if self.stop_vpi_grad:
            h = h.detach()
            if z is not None: z = z.detach()
        x = h
        b = x.shape[0]       
        if self.type_nn in [0, 1]:
            x_ = F.relu(self.conv1(x))
            x_ = F.relu(self.conv2(x_))
            x_ = torch.flatten(x_, start_dim=1)
            x_v, x_logits, x_done = x_, x_, x_
        elif self.type_nn in [2, 3]:
            for block in self.resblocks: x = block(x)
            if self.predict_v_pi:
                x_v = torch.flatten(nn.functional.relu(self.bn_v(self.conv1x1_v(x))), start_dim=1)
                x_logits = torch.flatten(nn.functional.relu(self.bn_logits(self.conv1x1_logits(x))), start_dim=1)
            if self.predict_done:
                x_done = torch.flatten(nn.functional.relu(self.bn_done(self.conv1x1_done(x))), start_dim=1)                
        elif self.type_nn in [4]:
            z = torch.flatten(z, start_dim=1)
            x = torch.concat([z, x], dim=1)
            x_v, x_logits, x_done = x, x, x

        if self.predict_v_pi:
            logits = self.fc_logits(x_logits)
            if self.reward_transform:
                v_enc_logit = self.fc_v(x_v)
                v_enc_v = F.softmax(v_enc_logit, dim=-1)
                _, v = self.reward_tran.decode(v_enc_v)
            else:
                v_enc_logit = None
                v = self.fc_v(x_v).squeeze(-1)
        else:
            v, v_enc_logit, logits = None, None, None

        if self.predict_done:
            done_logit = self.fc_done(x_done).squeeze(-1)
            done = (nn.Sigmoid()(done_logit) > 0.5).bool()
        else:
            done_logit, done = None, None

        if self.predict_r and predict_reward:
            if self.value_prefix:
                m = state[2] < self.max_unroll_length
                if torch.any(~m):
                    lstm_state = (state[0] * m.float().view(b, 1), state[1] * m.float().view(b, 1))                    
                    lstm_counter = state[2] * m.float()                            
                    last_r = state[3] * m.float()
                else:
                    lstm_state = (state[0], state[1])
                    lstm_counter = state[2] 
                    last_r = state[3]                
                if self.type_nn in [0, 1, 2, 3]:
                    x_r = self.conv1x1_reward(x)
                    x_r = self.bn_r_1(x_r)
                    x_r = nn.functional.relu(x_r)
                    x_r = x_r.view(b, self.lstm_input_size)
                elif self.type_nn in [4]:
                    x_r = x
                lstm_state = (lstm_state[0].unsqueeze(0), lstm_state[1].unsqueeze(0)) # the LSTM only has a single layer
                x_r, lstm_state = self.lstm(x_r.unsqueeze(0), lstm_state)
                lstm_state = (lstm_state[0].squeeze(0), lstm_state[1].squeeze(0))
                x_r = x_r.squeeze(0)

                state = lstm_state + (lstm_counter+1,)                
                x_r = self.bn_r_2(x_r)
                x_r = nn.functional.relu(x_r)
            else:
                if self.type_nn in [0, 1]:
                    x_r = x_
                elif self.type_nn in [2, 3]:
                    x_r = torch.flatten(nn.functional.relu(self.bn_r(self.conv1x1_r(x))), start_dim=1)
                elif self.type_nn in [4]:
                    x_r = x
            r_out = self.fc_r(x_r)
            if self.reward_transform:
                r_enc_logit = r_out
                r_enc_v = F.softmax(r_enc_logit, dim=-1)
                _, r = self.reward_tran.decode(r_enc_v)
            else:
                r_enc_logit = None
                r = r_out.squeeze(-1)
            if self.value_prefix:
                # if using value prefix, the r are the accumulating rewards;
                # so the reward for a single time step is the current accum. reward
                # minus the last accum reward
                single_r = r - last_r
                state = state + (r,)
            else:
                single_r = None
        else:
            single_r, r, r_enc_logit = None, None, None
        out = OutNetOut(single_rs=single_r, 
                         rs=r,
                         r_enc_logits=r_enc_logit, 
                         dones=done, 
                         done_logits=done_logit, 
                         vs=v, 
                         v_enc_logits=v_enc_logit, 
                         logits=logits, 
                         r_state=state)
        return out
    
    def init_state(self, bsz, device):
        if self.value_prefix:
            return (torch.zeros(bsz, 512, device=device), 
                    torch.zeros(bsz, 512, device=device), 
                    torch.zeros(bsz, device=device),
                    torch.zeros(bsz, device=device))
        else:
            return ()

class ModelNetBase(nn.Module):    
    def __init__(self, obs_shape, num_actions, flags):        
        super(ModelNetBase, self).__init__()      
        self.rnn = False
        self.flags = flags        
        self.obs_shape = obs_shape
        self.num_actions = num_actions          
        self.reward_transform = flags.reward_transform
        self.hz_tran = flags.model_hz_tran
        self.kl_alpha = flags.model_kl_alpha        
        self.type_nn = flags.model_type_nn # type_nn: type of neural network for the model; 0 for small, 1 for large, 2 for small enet, 3 for large enet
        self.size_nn = flags.model_size_nn # size_nn: int to adjust for size of model net (for model_type_nn == 3 only)
        self.frameEncoder = FrameEncoder(num_actions=num_actions, 
                                         input_shape=obs_shape, 
                                         type_nn=self.type_nn, 
                                         size_nn=self.size_nn,
                                         decoder=self.flags.model_img_loss_cost > 0.)
        f_shape = self.frameEncoder.out_shape
        inplanes = (f_shape[0] if self.type_nn not in [4] else 
                    f_shape[0] * f_shape[1])

        self.dynamicModel = DynamicModel(num_actions=num_actions, 
                                         inplanes=inplanes, 
                                         type_nn=self.type_nn, 
                                         size_nn=self.size_nn,
                                         disable_bn=self.flags.model_disable_bn)  
        d_outplanes = self.dynamicModel.outplanes
        
        input_shape = (f_shape if self.type_nn not in [4] else 
                    (f_shape[0] * f_shape[1] + d_outplanes,))
        
        self.output_rvpi = Output_rvpi(num_actions=num_actions, 
                                       input_shape=input_shape, 
                                       value_prefix=flags.value_prefix,
                                       max_unroll_length=flags.model_k_step_return, 
                                       reward_transform=self.reward_transform,
                                       stop_vpi_grad=flags.model_stop_vpi_grad, 
                                       zero_init=flags.model_zero_init, 
                                       type_nn=self.type_nn, 
                                       size_nn=self.size_nn,
                                       predict_done=self.flags.model_done_loss_cost > 0.,
                                       disable_bn=self.flags.model_disable_bn)
        
        if self.reward_transform:
            self.reward_tran = self.output_rvpi.reward_tran

        self.supervise = flags.model_sup_loss_cost > 0.
        self.model_supervise_type = flags.model_supervise_type

        if self.supervise:                        
            if self.model_supervise_type == 0:
                flatten_in_dim = (obs_shape[1]//16)*(obs_shape[2])//16*inplanes          
                self.P_1 = nn.Sequential(nn.Linear(flatten_in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                         nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                         nn.Linear(512, 1024), nn.BatchNorm1d(1024))
                self.P_2 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1024))
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        if self.hz_tran:
            if self.type_nn in [0, 1, 2, 3]:
                self.h_to_z_conv = nn.Sequential(
                    ResBlock(inplanes=inplanes, disable_bn=self.flags.model_disable_bn), 
                    conv3x3(inplanes, inplanes))
                self.z_to_h_conv = nn.Sequential(
                    ResBlock(inplanes=inplanes, disable_bn=self.flags.model_disable_bn), 
                    conv3x3(inplanes, inplanes))
            elif self.type_nn in [4]:
                self.h_to_z_mlp = mlp(d_outplanes, [d_outplanes], inplanes, 
                    activation=nn.ELU, norm=False)
                
        assert not (self.type_nn == 4 and not self.hz_tran), "dreamer net must be used with hz separation"

    def h_to_z(self, h, action=None, flatten=False):
        if not self.hz_tran: return h, None
        if flatten:
            h_ = torch.flatten(h, 0, 1)   
        else:
            h_ = h
        if self.type_nn in [4]: 
            b = h_.shape[0]
            s_size = 32 * self.size_nn
            z_logit = self.h_to_z_mlp(h).view(b, s_size, s_size)
            z_p = F.softmax(z_logit, dim=2)
            z = torch.multinomial(z_p.view(b*s_size, s_size), num_samples=1).view(b, s_size)
            z = F.one_hot(z, num_classes=s_size)
            z = z + z_p - z_p.detach()
        else:                        
            z = self.h_to_z_conv(h_)
            z_logit = None
        if flatten:
            z = z.view(h.shape[:2] + z.shape[1:])
            z_logit = z_logit.view(h.shape[:2] + z_logit.shape[1:]) if z_logit is not None else None            
        return z, z_logit
    
    def z_to_h(self, z, flatten=False):
        if self.type_nn in [4]: 
            in_shape = z.shape[0] if not flatten else (z.shape[0] * z.shape[1])
            h = self.dynamicModel.init_state(in_shape, device=z.device)[0]            
        else:
            if not self.hz_tran: return z
            if flatten:    
                z_ = torch.flatten(z, 0, 1)   
            else:
                z_ = z
            h = self.z_to_h_conv(z_)
        if flatten:
            h = h.view(z.shape[:2] + h.shape[1:])
        return h

    def supervise_loss(self, xs, model_net_out, is_weights, mask, one_hot=False):
        """
        Args:
            xs(tensor): state s with shape (k+1, B, *) in the form of s_t, s_{t+1}, ..., s_{t+k}
            model_net_out(tensor): model_net_out from forward
            mask(tensor): mask (float) with shape (k, B)
            im_weights(tensor): importance weight with shape (B) for each sample;  
        Return:
            loss(tensor): scalar self-supervised loss
        """
        k, b, *_ = xs.shape
        k = k - 1        
        
        if self.model_supervise_type in [0, 1, 2]:
            # 0 for SimSiam loss (efficient zero)
            # 1 for direct cos loss
            # 2 for direct L2 loss
            # 3 for dreamer loss
            true_zs = model_net_out.true_zs[1:]       
            pred_zs = torch.flatten(model_net_out.pred_zs[1:], 0, 1)        
            pred_zs = torch.flatten(pred_zs, 1)
            if self.model_supervise_type == 0:
                src = self.P_2(self.P_1(pred_zs))
            elif self.model_supervise_type in [1, 2]:
                src = pred_zs
            
            with torch.no_grad():
                true_zs = torch.flatten(true_zs, 0, 1)
                true_zs = torch.flatten(true_zs, 1)
                if self.model_supervise_type == 0:
                    tgt = self.P_1(true_zs)
                elif self.model_supervise_type in [1, 2]:
                    tgt = true_zs
            
            if self.model_supervise_type in [0, 1]:
                sup_loss = -self.cos(src, tgt.detach())
            elif self.model_supervise_type == 2:
                sup_loss = torch.mean((src - tgt.detach())**2, dim=-1)            
            sup_loss = sup_loss.view(k, b)
            s_mask = mask[1:]

        elif self.model_supervise_type in [3]:
            if self.flags.model_sup_ignore_first:
                true_z_logits = model_net_out.true_z_logits[1:]
                pred_z_logits = model_net_out.pred_z_logits[1:]
                k_ = k 
            else:
                true_z_logits = model_net_out.true_z_logits
                pred_z_logits = model_net_out.pred_z_logits
                k_ = k + 1

            _, _, c1, c2 = true_z_logits.shape
            true_z_logits = true_z_logits.reshape(k_*b*c1, c2)
            pred_z_logits = pred_z_logits.reshape(k_*b*c1, c2)

            alpha = self.kl_alpha
            target = F.softmax(true_z_logits, dim=-1)
            sup_loss_pre = torch.nn.CrossEntropyLoss(reduction="none")(
                    input = pred_z_logits,
                    target = target.detach())
            sup_loss_post = torch.nn.CrossEntropyLoss(reduction="none")(
                    input = pred_z_logits.detach(),
                    target = target)
            sup_loss = alpha  * sup_loss_pre + (1 - alpha) * sup_loss_post
            sup_loss = sup_loss.view(k_, b, c1)   
            sup_loss = torch.sum(sup_loss, dim=-1)

            if self.flags.model_sup_ignore_first:
                s_mask = mask[1:]
            else:                
                s_mask = mask
                
        if mask is not None: sup_loss = sup_loss * s_mask     
        sup_loss = torch.sum(sup_loss, dim=0)    
        sup_loss = sup_loss * is_weights
        sup_loss = torch.sum(sup_loss)

        return sup_loss

    def img_loss(self, xs, model_net_out, is_weights, mask):
        """
        Args:
            xs(tensor): state s with shape (k+1, B, *) in the form of s_t, s_{t+1}, ..., s_{t+k}
            model_net_out(tensor): model_net_out from forward
            mask(tensor): mask (float) with shape (k, B)
            im_weights(tensor): importance weight with shape (B) for each sample;  
        Return:
            loss(tensor): scalar img reconstruction loss
        """    
        k, b, *_ = xs.shape
        k = k - 1    

        if self.flags.model_img_loss_cost > 0:
            # image reconstruction loss

            if self.flags.model_sup_ignore_first:
                true_zs = model_net_out.true_zs[1:]
                pred_zs = model_net_out.pred_zs[1:]
                hs = model_net_out.hs[1:]
                xs = xs[1:]
            else:
                true_zs = model_net_out.true_zs
                pred_zs = model_net_out.pred_zs
                hs = model_net_out.hs 
                xs = xs

            if self.flags.model_img_loss_use_pred_zs:
                decoder_in = pred_zs
            else:
                decoder_in = true_zs

            if self.type_nn in [4]:
                h = torch.flatten(hs, 0, 1)
            else:
                h = None

            pred_xs = self.frameEncoder.decode(torch.flatten(decoder_in, 0, 1), h)
            xs = torch.flatten(xs, 0, 1).float() / 255.0
            img_loss = torch.sum(torch.square(xs - pred_xs), dim=(1, 2, 3))                        
            img_loss = img_loss.view(k if self.flags.model_sup_ignore_first else k+1, b)

            if mask is not None: 
                if self.flags.model_sup_ignore_first:
                    i_mask = mask[1:]
                else:
                    i_mask = mask
                img_loss = img_loss * i_mask

            img_loss = torch.sum(img_loss, dim=0)
            img_loss = img_loss * is_weights
            img_loss = torch.sum(img_loss)
        else:
            img_loss = None

        return img_loss
        
    def forward(self, xs, actions, one_hot=False, compute_true_z=False, inference=True):
        """
        Args:
            x(tensor): frames (uint8) with shape (k+1, B, C, H, W), in the form of s_t, s_{t+1}, ..., s_{t+k}, or
                with shape (B, C, H, W) in the form of s_t; 
            actions(tensor): action (int64) with shape (k+1, B), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding      
            true_z(bool): if True, true z will be generated (require x to be in shape (k+1, ...))
            inference(bool): if True, predicted z will be fed back into the network, else true z
                 (require x to be in shape (k+1, ...))
        Return:
            rs(tensor): predicted reward with shape (k, B, ...), in the form of r_{t+1}, r_{t+2}, ..., r_{t+k}
            vs(tensor): predicted value with shape (k+1, B, ...), in the form of v_{t}, v_{t+1}, v_{t+2}, ..., v_{t+k}
            logits(tensor): predicted policy with shape (k+1, B, ...), in the form of pi_{t}, pi_{t+1}, pi_{t+2}, ..., pi_{t+k}
            pred_zs(tensor): predicted encoded states with shape (k+1, B, ...), in the form of z_t, z_{t+1}, z_{t+2}, ..., z_{t+k}
            true_zs(tensor): true encoded states with shape (k+1, B, ...), in the form of z_t, z_{t+1}, z_{t+2}, ..., z_{t+k}
            hs(tensor): hidden state with shape (k+1, B, ...), in the form of h_t, h_{t+1}, h_{t+2}, ..., h_{t+k}
            r_state(tensor): reward hidden state with shape (B, ...)
            (Recall we use the transition notation: s_t, a_t, r_{t+1}, s_{t+1}, ...)                
        """        
        k, b, *_ = actions.shape
        k = k - 1
        if len(xs.shape) == 4: xs = xs.unsqueeze(0)
        if compute_true_z or not inference:
            assert xs.shape[0] == k + 1, "in non-inference or true_z mode, xs shape should be k+1 instead of %d" % xs.shape[0]

        # initialise empty list
        data = {key:[] for key in ModelNetOut._fields}
        
        if not one_hot: actions = F.one_hot(actions, self.num_actions)  
        if self.type_nn in [0, 1, 2, 3]:
            true_z, true_z_logit = self.frameEncoder(xs[0], actions[0], None)    
            h = self.z_to_h(true_z, flatten=False)
        elif self.type_nn in [4]:
            h = self.dynamicModel.init_state(bsz=b, device=xs.device)[0]
            true_z, true_z_logit = self.frameEncoder(xs[0], actions[0], h)  
        r_state = self.output_rvpi.init_state(bsz=xs.shape[1], device=xs.device)
        out_net_out = self.output_rvpi(true_z, h, predict_reward=False, state=())        
        pred_z, pred_z_logit = self.h_to_z(h)

        data["vs"].append(out_net_out.vs)
        data["v_enc_logits"].append(out_net_out.v_enc_logits)
        data["logits"].append(out_net_out.logits)
        data["pred_zs"].append(pred_z)
        data["pred_z_logits"].append(pred_z_logit)
        data["true_zs"].append(true_z)
        data["true_z_logits"].append(true_z_logit)
        data["hs"].append(h)

        z_in = true_z

        for t in range(1, actions.shape[0]):  
            out = self.forward_zh(z_in, h, r_state, actions[t])
            for key in out._fields:
                if key in ["true_zs", "true_z_logits", "r_state"]: continue
                val = getattr(out, key)
                if val is not None: val = val.squeeze(0)
                data[key].append(val)
            if compute_true_z or not inference:
                true_z, true_z_logit = self.frameEncoder(xs[t], actions[t], data["hs"][t])    
                data["true_zs"].append(true_z)
                data["true_z_logits"].append(true_z_logit)
            z_in = out.pred_zs[-1] if inference else true_z
            h = out.hs[-1]
            r_state = out.r_state
        
        for key in data.keys():
            if len(data[key]) > 0 and data[key][0] is not None:
                data[key] = torch.concat([val.unsqueeze(0) for val in data[key]], dim=0)
            else:
                data[key] = None
        
        data["r_state"] = r_state
        return util.construct_tuple(ModelNetOut, **data)
    
    def forward_zh(self, z, h, r_state, action, one_hot=True):
        """
        One-step transition from z_t, h_t, a_t to predicted z_{t+1}, h_{t+1}, r_{t+1}, v_{t+1}, pi_{t+1}
        Args:
            z: encoded state with shape (B, ...), in the form of z_t
            h: hidden state with shape (B, ...), in the form of h_t
            r_state: hidden state of reward predictor with shape (B, ...)
            action: action with shape (B, ...)
        """
        if not one_hot: action = F.one_hot(action, self.num_actions)                  
        h = self.dynamicModel(z=z, h=h, actions=action)        
        pred_z, pred_z_logit = self.h_to_z(h, action=action)
        out_net_out = self.output_rvpi(
                z, h, predict_reward=True, state=r_state)

        return ModelNetOut(single_rs=util.safe_unsqueeze(out_net_out.single_rs, 0),
                           rs=util.safe_unsqueeze(out_net_out.rs, 0), 
                           r_enc_logits=util.safe_unsqueeze(out_net_out.r_enc_logits, 0), 
                           dones=util.safe_unsqueeze(out_net_out.dones, 0), 
                           done_logits=util.safe_unsqueeze(out_net_out.done_logits, 0), 
                           vs=util.safe_unsqueeze(out_net_out.vs, 0), 
                           v_enc_logits=util.safe_unsqueeze(out_net_out.v_enc_logits, 0), 
                           logits=util.safe_unsqueeze(out_net_out.logits, 0), 
                           pred_xs=None,
                           pred_zs=util.safe_unsqueeze(pred_z, 0),
                           pred_z_logits=util.safe_unsqueeze(pred_z_logit, 0),
                           true_zs=None,
                           true_z_logits=None,
                           hs=util.safe_unsqueeze(h, 0), 
                           r_state=r_state,
                           )
    
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        device = next(self.parameters()).device
        if device != torch.device("cpu"):
            weights = {k: v.to(device) for k, v in weights.items()}
        self.load_state_dict(weights)        

class DuelNetBase(nn.Module):    
    def __init__(self, obs_shape, num_actions, flags):        
        super(DuelNetBase, self).__init__()      
        self.rnn = False
        self.flags = flags        
        self.obs_shape = obs_shape
        self.num_actions = num_actions          
        self.reward_transform = flags.reward_transform
        self.type_nn = flags.model_type_nn # type_nn: type of neural network for the model; 0 for small, 1 for large, 2 for small enet, 3 for large enet
        self.size_nn = flags.model_size_nn # size_nn: int to adjust for size of model net (for model_type_nn == 3 only)
        self.debug = False
        self.debug_n = 0

        self.modelEncoder = FrameEncoder(num_actions=num_actions, 
                                         input_shape=obs_shape, 
                                         type_nn=self.type_nn, 
                                         size_nn=self.size_nn,
                                         decoder=True)
        self.hidden_shape = self.modelEncoder.out_shape
        inplanes = self.hidden_shape[0]
        self.modelRNN = DynamicModel(num_actions=num_actions, 
                                     inplanes=inplanes, 
                                     type_nn=self.type_nn, 
                                     size_nn=self.size_nn,
                                     disable_half_grad=True,
                                     disable_bn=self.flags.model_disable_bn)  
        self.modelOut = Output_rvpi(num_actions=num_actions, 
                                       input_shape=self.hidden_shape, 
                                       value_prefix=flags.value_prefix,
                                       max_unroll_length=flags.model_k_step_return, 
                                       reward_transform=self.reward_transform,
                                       stop_vpi_grad=False, 
                                       zero_init=flags.model_zero_init, 
                                       type_nn=self.type_nn, 
                                       size_nn=self.size_nn,
                                       predict_v_pi=False,
                                       predict_r=True,
                                       predict_done=self.flags.model_done_loss_cost > 0.,
                                       disable_bn=self.flags.model_disable_bn)
        
        self.predEncoder = FrameEncoder(num_actions=num_actions, 
                                        input_shape=obs_shape, 
                                        type_nn=self.type_nn, 
                                        size_nn=self.size_nn,
                                        decoder=False)
        self.predRNN = DynamicModel(num_actions=num_actions, 
                                    inplanes=inplanes*2, 
                                    outplanes=inplanes,
                                    type_nn=self.type_nn, 
                                    size_nn=self.size_nn,
                                    disable_half_grad=False,
                                    disable_bn=self.flags.model_disable_bn)    
        self.predOut = Output_rvpi(num_actions=num_actions, 
                                   input_shape=self.hidden_shape, 
                                   value_prefix=flags.value_prefix,
                                   max_unroll_length=flags.model_k_step_return, 
                                   reward_transform=self.reward_transform,
                                   stop_vpi_grad=False, 
                                   zero_init=flags.model_zero_init, 
                                   type_nn=self.type_nn, 
                                   size_nn=self.size_nn,
                                   predict_v_pi=True,
                                   predict_r=False,
                                   predict_done=False,
                                   disable_bn=self.flags.model_disable_bn)
        
        if self.reward_transform:
            self.reward_tran = self.predOut.reward_tran
        self.sep_h_idx = inplanes
    
    def forward(self, xs, actions, one_hot=False, compute_true_z=False, inference=True):
        """
        Args:
            x(tensor): frames (uint8) with shape (k+1, B, C, H, W), in the form of s_t, s_{t+1}, ..., s_{t+k}, or
                with shape (B, C, H, W) in the form of s_t; 
            actions(tensor): action (int64) with shape (k+1, B), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding      
            true_z(bool): if True, true z will be generated (require x to be in shape (k+1, ...))
            inference(bool): if True, predicted z will be fed back into the network, else true z
                 (require x to be in shape (k+1, ...))
        Return:
            rs(tensor): predicted reward with shape (k, B, ...), in the form of r_{t+1}, r_{t+2}, ..., r_{t+k}
            vs(tensor): predicted value with shape (k+1, B, ...), in the form of v_{t}, v_{t+1}, v_{t+2}, ..., v_{t+k}
            logits(tensor): predicted policy with shape (k+1, B, ...), in the form of pi_{t}, pi_{t+1}, pi_{t+2}, ..., pi_{t+k}
            pred_zs(tensor): predicted encoded states with shape (k+1, B, ...), in the form of z_t, z_{t+1}, z_{t+2}, ..., z_{t+k}
            true_zs(tensor): true encoded states with shape (k+1, B, ...), in the form of z_t, z_{t+1}, z_{t+2}, ..., z_{t+k}
            hs(tensor): hidden state with shape (k+1, B, ...), in the form of h_t, h_{t+1}, h_{t+2}, ..., h_{t+k}
            r_state(tensor): reward hidden state with shape (B, ...)
            (Recall we use the transition notation: s_t, a_t, r_{t+1}, s_{t+1}, ...)                
        """     
        if self.debug and len(xs.shape) == 5 and xs.shape[0] > 1:
            import numpy as np
            np.save("/home/sc/RS/thinker/test/tmp/xs.npy", xs.detach().cpu().numpy())
            np.save("/home/sc/RS/thinker/test/tmp/as.npy", actions.detach().cpu().numpy())
        k, b, *_ = actions.shape
        k = k - 1
        if len(xs.shape) == 4: xs = xs.unsqueeze(0)
        if compute_true_z or not inference:
            assert xs.shape[0] == k + 1, "in non-inference or true_z mode, xs shape should be k+1 instead of %d" % xs.shape[0]
        if not one_hot: actions = F.one_hot(actions, self.num_actions) 
                
        model_h, _ = self.modelEncoder(xs[0], actions[0], None)         
        model_hs = [model_h.unsqueeze(0)]        
        for t in range(1, k+1):              
            model_h = self.modelRNN(z=None, h=model_h, actions=actions[t])       
            model_hs.append(model_h.unsqueeze(0))
        model_hs = torch.concat(model_hs, dim=0)        
        model_xs = self.modelEncoder.decode(model_hs[1:], flatten=True)  

        model_out_net_outs = []
        r_state = self.modelOut.init_state(bsz=xs.shape[1], device=xs.device)    
        for t in range(1, k+1):      
            model_out_net_out = self.modelOut(
                None, model_hs[t], predict_reward=True, state=r_state)
            model_out_net_outs.append(model_out_net_out)
            r_state = model_out_net_out.r_state

        pred_in_xs = torch.concat([xs[[0]].float()/255., model_xs], dim=0)
        pred_zs, _ = self.predEncoder(pred_in_xs.detach(), actions, flatten=True, rescale=False) 
        pred_h = torch.zeros(size=(b,) + self.hidden_shape, device=xs.device)
        predRNN_in = torch.concat([pred_h, pred_zs[0]], dim=1)
        pred_h = self.predRNN(z=None, h=predRNN_in, actions=actions[0])        
        pred_hs = [pred_h.unsqueeze(0)]
        for t in range(1, k+1): 
            predRNN_in = torch.concat([pred_h, pred_zs[t]], dim=1)
            pred_h = self.predRNN(z=None, h=predRNN_in, actions=actions[t])       
            pred_hs.append(pred_h.unsqueeze(0))
        pred_hs = torch.concat(pred_hs, dim=0)     

        pred_out_net_outs = []
        for t in range(0, k+1):      
            pred_out_net_out = self.predOut(None, 
                pred_hs[t], predict_reward=False, state=())        
            pred_out_net_outs.append(pred_out_net_out) 

        hs = torch.concat([model_hs, pred_hs], dim=2)

        if self.flags.actor_only_see_h: 
            true_zs = pred_hs[[0]]
        else:
            true_zs = pred_zs[[0]]

        model_net_out = ModelNetOut(
            single_rs=util.safe_concat(model_out_net_outs, "single_rs", 0),
            rs=util.safe_concat(model_out_net_outs, "rs", 0),
            r_enc_logits=util.safe_concat(model_out_net_outs, "r_enc_logits", 0), 
            dones=util.safe_concat(model_out_net_outs, "dones", 0), 
            done_logits=util.safe_concat(model_out_net_outs, "done_logits", 0), 
            vs=util.safe_concat(pred_out_net_outs, "vs", 0), 
            v_enc_logits=util.safe_concat(pred_out_net_outs, "v_enc_logits", 0), 
            logits=util.safe_concat(pred_out_net_outs, "logits", 0), 
            pred_xs=model_xs,
            pred_zs=pred_zs,
            pred_z_logits=None,
            true_zs=true_zs,
            true_z_logits=None,
            hs=hs, 
            r_state=r_state,
            )
        if self.debug: 
            self.model_net_out = model_net_out
            if model_net_out.pred_xs is not None:            
                #util.plot_gym_env_out(model_net_out.pred_xs[-1])
                savepath = "/home/sc/RS/thinker/test/tmp/"
                for t in range(model_net_out.pred_xs.shape[0]):
                    util.plot_gym_env_out(model_net_out.pred_xs[t], title="B_%d_%d pred_x" % (self.debug_n, t), savepath=savepath)
                self.debug_n += 1
        return model_net_out    

    def forward_zh(self, z, h, r_state, action, one_hot=True):
        """
        One-step transition from z_t, h_t, a_t to predicted z_{t+1}, h_{t+1}, r_{t+1}, v_{t+1}, pi_{t+1}
        Args:
            z: encoded state with shape (B, ...), in the form of z_t
            h: hidden state with shape (B, ...), in the form of h_t
            r_state: hidden state of reward predictor with shape (B, ...)
            action: action with shape (B, ...)
        """
        if not one_hot: action = F.one_hot(action, self.num_actions)                  
        model_h, pred_h = h[:, :self.sep_h_idx], h[:, self.sep_h_idx:]

        model_h = self.modelRNN(z=None, h=model_h, actions=action)     
        model_x = self.modelEncoder.decode(model_h)   
        model_out_net_out = self.modelOut(
                None, model_h, predict_reward=True, state=r_state)
        
        pred_z, _ = self.predEncoder(model_x.detach(), action, rescale=False)
        predRNN_in = torch.concat([pred_h, pred_z], dim=1)
        pred_h = self.predRNN(z=None, h=predRNN_in, actions=action)          
        pred_out_net_out = self.predOut(
                None, pred_h, predict_reward=False, state=None)  
        h = torch.concat([model_h, pred_h], dim=1)
        if self.flags.actor_only_see_h: pred_z = pred_h
        #util.plot_gym_env_out(model_x)

        return ModelNetOut(single_rs=util.safe_unsqueeze(model_out_net_out.single_rs, 0),
                           rs=util.safe_unsqueeze(model_out_net_out.rs, 0), 
                           r_enc_logits=util.safe_unsqueeze(model_out_net_out.r_enc_logits, 0), 
                           dones=util.safe_unsqueeze(model_out_net_out.dones, 0), 
                           done_logits=util.safe_unsqueeze(model_out_net_out.done_logits, 0), 
                           vs=util.safe_unsqueeze(pred_out_net_out.vs, 0), 
                           v_enc_logits=util.safe_unsqueeze(pred_out_net_out.v_enc_logits, 0), 
                           logits=util.safe_unsqueeze(pred_out_net_out.logits, 0), 
                           pred_xs=util.safe_unsqueeze(model_x, 0),
                           pred_zs=util.safe_unsqueeze(pred_z, 0),
                           pred_z_logits=None,
                           true_zs=None,
                           true_z_logits=None,
                           hs=util.safe_unsqueeze(h, 0), 
                           r_state=r_state,
                           )
    
    def img_loss(self, xs, model_net_out, is_weights, mask, one_hot=False):
        k, b, *_ = xs.shape
        k = k - 1 

        if self.debug:
        #if True:
            for t in range(k):
                savepath = "/home/sc/RS/thinker/test/tmp/"
                util.plot_gym_env_out(xs[t+1].float()/255., title="%d_%d_true_x" % (
                    self.debug_n, t), savepath=savepath)
                util.plot_gym_env_out(model_net_out.pred_xs[t], title="%d_%d_pred_x" % (
                    self.debug_n, t), savepath=savepath)
            self.debug_n += 1

        pred_xs = torch.flatten(model_net_out.pred_xs, 0, 1)
        xs = torch.flatten(xs[1:], 0, 1).float() / 255.0
        img_loss = torch.mean(torch.square(xs - pred_xs), dim=(1, 2, 3))                        
        img_loss = img_loss.view(k, b)
        if mask is not None: img_loss = img_loss * mask[1:]
        img_loss = torch.sum(img_loss, dim=0)
        img_loss = img_loss * is_weights
        img_loss = torch.sum(img_loss)

        return img_loss        
    
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        device = next(self.parameters()).device
        if device != torch.device("cpu"):
            weights = {k: v.to(device) for k, v in weights.items()}
        self.load_state_dict(weights)     

def ModelNet(obs_shape, num_actions, flags):
    if flags.duel_net:
        return DuelNetBase(obs_shape, num_actions, flags)
    else:
        return ModelNetBase(obs_shape, num_actions, flags)
    
class RewardTran(nn.Module):    
    def __init__(self, vec, support=300, eps=0.001):
        super(RewardTran, self).__init__()
        self.support = support
        self.eps = eps
        self.vec = vec
        if self.vec:
            self.dec = torch.arange(-support, support+1,1)        
            self.encoded_n = 2 * self.support + 1
            self.register_buffer('dec_const', self.dec)

    def forward(self, x):
        """encode the unencoded scalar reward or values to encoded scalar (and encoded vector) according to MuZero"""
        with torch.no_grad():
            sup, eps = self.support, self.eps
            enc_s = torch.sign(x)*(torch.sqrt(torch.abs(x)+1)-1)+eps*x
            if not self.vec: return enc_s
            enc_s = torch.clamp(enc_s, -sup, +sup)            
            enc_v = torch.zeros(enc_s.shape+(2*sup+1,), dtype=torch.float32, device=enc_s.device)        
            enc_s_floor = torch.floor(enc_s)
            enc_v_reminder = enc_s - enc_s_floor
            enc_s_floor = enc_s_floor.long().unsqueeze(-1)
            enc_v.scatter_(-1, torch.clamp_max(sup+enc_s_floor+1, 2*sup) , enc_v_reminder.unsqueeze(-1))
            enc_v.scatter_(-1, sup+enc_s_floor, 1-enc_v_reminder.unsqueeze(-1))        
            return enc_s, enc_v

    def encode(self, x):
        return self.forward(x)

    def decode(self, x):
        """dncode the encoded vector (or encoded scalar) to unencoded scalar (and encoded scalar) according to MuZero"""    
        with torch.no_grad():  
            eps = self.eps     
            if self.vec:            
                enc_s = torch.sum(self.dec_const*x, dim=-1)   
            else:
                enc_s = x
            dec_s = torch.sign(enc_s)*(torch.square((torch.sqrt(1+4*eps*(torch.abs(enc_s)+1+eps))-1)/(2*eps)) - 1) 
            if self.vec:
                return enc_s, dec_s    
            else:
                return dec_s