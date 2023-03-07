from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from thinker.core.rnn import ConvAttnLSTM

ActorOut = namedtuple('ActorOut', ['policy_logits', 'im_policy_logits', 'reset_policy_logits', 
    'action', 'im_action', 'reset_action', 'baseline', 'baseline_enc_s', 'reg_loss'])
ModelNetOut = namedtuple('ModelNetOut', ['rs', 'r_enc_logits', 'vs', 'v_enc_logits', 'logits', 
                                         'encodes', 'state', 'single_rs'])

def add_hw(x, h, w):
    return x.unsqueeze(-1).unsqueeze(-1).broadcast_to(x.shape + (h,w))

def mlp(input_size, layer_sizes, output_size, output_activation=nn.Identity, activation=nn.ReLU,
    momentum=0.1, zero_init=False):
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
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
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

    def __init__(self, inplanes, outplanes=None, stride=1, downsample=None):
        super().__init__()
        if outplanes is None: outplanes = inplanes 
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
            self.frame_encoder = FrameEncoder(frame_channels=input_shape[0], num_actions=num_actions, 
                    down_scale_c=down_scale_c, concat_action=False)
            in_channels = 256 // down_scale_c
            in_hw = input_shape[1] // 16 
        else:
            in_channels = input_shape[0]
            in_hw = input_shape[1]

        # input shape here is (in_channels, in_hw, in_hw)
        if not drc:
            self.conv = torch.nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=encoder_out_channels*2, kernel_size=3, padding='same'), 
                nn.ReLU(), 
                nn.Conv2d(in_channels=encoder_out_channels*2, out_channels=encoder_out_channels, kernel_size=3, padding='same'), 
                nn.ReLU())            
        else:
            self.conv = torch.nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=encoder_out_channels, kernel_size=3, padding='same'), 
                nn.ReLU())
            self.drc_core = ConvAttnLSTM(h=in_hw, w=in_hw,
                                        input_dim=encoder_out_channels, hidden_dim=encoder_out_channels, kernel_size=3, num_layers=3, 
                                        num_heads=8, mem_n=0, attn=False, attn_mask_b=0.,
                                        grad_scale=self.rnn_grad_scale)                   
        
        if self.compress:            
            self.fc = torch.nn.Sequential(nn.Linear(encoder_out_channels * in_hw * in_hw, compress_size), nn.ReLU())

        self.out_size = (encoder_out_channels * in_hw * in_hw) if not self.compress else compress_size
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
            x = self.frame_encoder(x, actions=None)
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
                in_channels= 128 if self.model_type_nn in [0, 1, 3] else 64
                input_shape = (in_channels, gym_obs_shape[1]//16, gym_obs_shape[2]//16)
            compress_size = 128 if self.actor_encode_concat_type == 1 else 256
            self.actor_encoder = ActorEncoder(input_shape=input_shape, num_actions=num_actions,
                frame_encode=not self.actor_see_encode, double_encode=self.actor_see_double_encode, 
                compress_size=compress_size, drc=self.actor_drc, rnn_grad_scale=self.rnn_grad_scale)

        in_channels = self.obs_shape[0]
        if self.actor_see_p > 0 and self.actor_encode_concat_type == 1:
            in_channels = in_channels + self.actor_encoder.out_size

        self.initial_enc = torch.nn.Sequential(nn.Linear(in_channels, self.tran_dim), 
                                              nn.ReLU(),
                                              nn.Linear(self.tran_dim, self.tran_dim), 
                                              nn.ReLU())

        self.core = ConvAttnLSTM(h=1, w=1,
                                input_dim=self.tran_dim, hidden_dim=self.tran_dim,
                                kernel_size=1, num_layers=self.tran_layer_n,
                                num_heads=8, mem_n=self.tran_mem_n, attn=not self.tran_lstm_no_attn,
                                attn_mask_b=self.attn_mask_b, grad_scale=self.rnn_grad_scale)                         
        
        last_out_size = 256           
        self.fc = torch.nn.Sequential(nn.Linear(self.tran_dim, last_out_size),  nn.ReLU())              
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

DOWNSCALE_C = 2   
class FrameEncoder(nn.Module):    
    def __init__(self, num_actions, frame_channels=3, type_nn=0, size_nn=1, down_scale_c=2, concat_action=True):
        super(FrameEncoder, self).__init__() 
        self.num_actions = num_actions
        self.type_nn = type_nn
        self.size_nn = size_nn
        self.down_scale_c=down_scale_c
        if type_nn in [0, 1]:
            self.concat_action=concat_action
        elif type_nn in [2, 3]:
            self.concat_action=False        

        if self.concat_action:
            in_channels=frame_channels+num_actions
        else:
            in_channels=frame_channels        

        if type_nn in [0, 1]:
        
            if type_nn == 0:
                n_block = 1
            elif type_nn == 1:
                n_block = 2

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128//down_scale_c, kernel_size=3, stride=2, padding=1) 
            res = nn.ModuleList([ResBlock(inplanes=128//down_scale_c) for i in range(n_block)]) # Deep: 2 blocks here
            self.res1 = torch.nn.Sequential(*res)
            self.conv2 = nn.Conv2d(in_channels=128//down_scale_c, out_channels=256//down_scale_c, 
                                kernel_size=3, stride=2, padding=1) 
            res = nn.ModuleList([ResBlock(inplanes=256//down_scale_c) for i in range(n_block)]) # Deep: 3 blocks here
            self.res2 = torch.nn.Sequential(*res)
            self.avg1 = nn.AvgPool2d(3, stride=2, padding=1)
            res = nn.ModuleList([ResBlock(inplanes=256//down_scale_c) for i in range(n_block)]) # Deep: 3 blocks here
            self.res3 = torch.nn.Sequential(*res)
            self.avg2 = nn.AvgPool2d(3, stride=2, padding=1)

        elif type_nn in [2, 3]:
            # efficient zero
            num_block = 1 if type_nn == 2 else (2 * self.size_nn)
            out_channels = 64 if type_nn == 2 else 128
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
    
    def forward(self, x, actions):      
        """
        Args:
          x (tensor): frame with shape B, C, H, W        
          action (tensor): action with shape B, num_actions (in one-hot)
        """
        
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
        return x
    
class DynamicModel(nn.Module):
    def __init__(self, num_actions, inplanes, type_nn=0, size_nn=1):        
        super(DynamicModel, self).__init__()
        self.num_actions = num_actions     
        self.type_nn = type_nn
        self.size_nn = size_nn        

        if type_nn == 0:
            res = nn.ModuleList([ResBlock(inplanes=inplanes+num_actions, outplanes=inplanes)] + [
                    ResBlock(inplanes=inplanes) for i in range(4)]) 
            self.res = torch.nn.Sequential(*res)
        elif type_nn == 1:                      
            res = nn.ModuleList([ResBlock(inplanes=inplanes) for i in range(15)] + [
                    ResBlock(inplanes=inplanes, outplanes=inplanes*num_actions)])                    
        elif type_nn in [2, 3]:
            num_block = 1 if type_nn == 2 else (4 * self.size_nn)
            self.conv1 =  conv3x3(inplanes+num_actions, inplanes)
            self.bn1 = nn.BatchNorm2d(inplanes)
            res = nn.ModuleList([ResBlock(inplanes=inplanes) for i in range(num_block)])
        self.res = torch.nn.Sequential(*res)    
    
    def forward(self, x, actions):              
        bsz, c, h, w = x.shape
        if self.training:
            x.register_hook(lambda grad: grad * 0.5)
        if self.type_nn == 0:
            actions = actions.unsqueeze(-1).unsqueeze(-1).tile  ([1, 1, x.shape[2], x.shape[3]])        
            x = torch.concat([x, actions], dim=1)
            out = self.res(x)
        elif self.type_nn == 1:            
            res_out = self.res(x).view(bsz, self.num_actions, c, h, w)        
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
        return out
    
class Output_rvpi(nn.Module):   
    def __init__(self, num_actions, input_shape, value_prefix, max_unroll_length, 
            reward_transform, zero_init, type_nn, size_nn):         
        super(Output_rvpi, self).__init__()        
        c, h, w = input_shape
        self.input_shape = input_shape
        self.type_nn = type_nn
        self.size_nn = size_nn
        self.value_prefix = value_prefix
        self.max_unroll_length = max_unroll_length
        self.reward_transform = reward_transform

        if self.type_nn in [0, 1]:
            self.conv1 = nn.Conv2d(in_channels=c, out_channels=c//2, kernel_size=3, padding='same') 
            self.conv2 = nn.Conv2d(in_channels=c//2, out_channels=c//4, kernel_size=3, padding='same') 
            fc_in = h * w * (c // 4)        
            self.fc_logits = nn.Linear(fc_in, num_actions)         

            if self.reward_transform:
                self.reward_tran = RewardTran(vec=True)
                self.fc_v = nn.Linear(fc_in, self.reward_tran.encoded_n) 
                if not self.value_prefix:
                    self.fc_r = nn.Linear(fc_in, self.reward_tran.encoded_n)         
            else:
                self.fc_v = nn.Linear(fc_in, 1) 
                if not self.value_prefix:
                    self.fc_r = nn.Linear(fc_in, 1)         

            if zero_init:
                torch.nn.init.constant_(self.fc_v.weight, 0.)
                torch.nn.init.constant_(self.fc_v.bias, 0.)
                torch.nn.init.constant_(self.fc_logits.weight, 0.)
                torch.nn.init.constant_(self.fc_logits.bias, 0.)
                if not self.value_prefix:
                    torch.nn.init.constant_(self.fc_r.weight, 0.)
                    torch.nn.init.constant_(self.fc_r.bias, 0.)

        elif self.type_nn in [2, 3]:
            num_block = 1 if self.type_nn  == 2 else 2 * self.size_nn
            out_channels = 16 if self.type_nn  == 2 else 16 * self.size_nn
            self.resblocks = nn.ModuleList(
                [ResBlock(c, c) for _ in range(num_block)]
            )
            self.conv1x1_v = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1)
            self.conv1x1_logits = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1)
            self.bn_v = nn.BatchNorm2d(out_channels)            
            self.bn_logits = nn.BatchNorm2d(out_channels)
            if self.reward_transform:
                self.reward_tran = RewardTran(vec=True)
                out_n = self.reward_tran.encoded_n
            else:
                out_n = 1            
            self.fc_v = mlp(h * w * out_channels, [32], out_n, zero_init=zero_init)            
            self.fc_logits = mlp(h * w * out_channels, [32], num_actions, zero_init=zero_init)                                  
            if not self.value_prefix:
                self.conv1x1_r = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1)
                self.bn_r = nn.BatchNorm2d(out_channels)
                self.fc_r = mlp(h * w * out_channels, [32], out_n, zero_init=zero_init)
            
        if self.value_prefix:
            self.conv1x1_reward = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=1)            
            self.bn_r_1 = nn.BatchNorm2d(16)
            self.lstm_input_size=16*input_shape[1]*input_shape[2]
            self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=512)
            self.bn_r_2 = nn.BatchNorm1d(512)
            out_n = self.reward_tran.encoded_n if self.reward_transform else 1
            self.fc_r = mlp(512, [64], out_n, zero_init=zero_init)                
        
    def forward(self, x, predict_reward=True, state=()):   
        b = x.shape[0]         
        if self.type_nn in [0, 1]:
            x_ = F.relu(self.conv1(x))
            x_ = F.relu(self.conv2(x_))
            x_ = torch.flatten(x_, start_dim=1)
            x_v, x_logits = x_, x_
        elif self.type_nn in [2, 3]:
            for block in self.resblocks: x = block(x)
            x_v = torch.flatten(nn.functional.relu(self.bn_v(self.conv1x1_v(x))), start_dim=1)
            x_logits = torch.flatten(nn.functional.relu(self.bn_logits(self.conv1x1_logits(x))), start_dim=1)
        logits = self.fc_logits(x_logits)
        if self.reward_transform:
            v_enc_logits = self.fc_v(x_v)
            v_enc_v = F.softmax(v_enc_logits, dim=-1)
            _, v = self.reward_tran.decode(v_enc_v)
        else:
            v_enc_logits = None
            v = self.fc_v(x_v).squeeze(-1)

        if predict_reward:
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
                x_r = self.conv1x1_reward(x)
                x_r = self.bn_r_1(x_r)
                x_r = nn.functional.relu(x_r)

                x_r = x_r.view(b, self.lstm_input_size).unsqueeze(0)
                lstm_state = (lstm_state[0].unsqueeze(0), lstm_state[1].unsqueeze(0)) # the LSTM only has a single layer
                x_r, lstm_state = self.lstm(x_r, lstm_state)
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
            r_out = self.fc_r(x_r)
            if self.reward_transform:
                r_enc_logits = r_out
                r_enc_v = F.softmax(r_enc_logits, dim=-1)
                _, r = self.reward_tran.decode(r_enc_v)
            else:
                r_enc_logits = None
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
            single_r, r, r_enc_logits = None, None, None
        return single_r, r, r_enc_logits, v, v_enc_logits, logits, state
    
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
        self.type_nn = flags.model_type_nn # type_nn: type of neural network for the model; 0 for small, 1 for large, 2 for small enet, 3 for large enet
        self.size_nn = flags.model_size_nn # size_nn: int to adjust for size of model net (for model_type_nn == 3 only)
        self.frameEncoder = FrameEncoder(num_actions=num_actions, frame_channels=obs_shape[0], type_nn=self.type_nn, size_nn=self.size_nn)
        if self.type_nn in [0, 1]:
            inplanes = 256 // DOWNSCALE_C
        elif self.type_nn == 2:
            inplanes = 64
        elif self.type_nn == 3:
            inplanes = 128 
        self.dynamicModel = DynamicModel(num_actions=num_actions, 
                                         inplanes=inplanes, 
                                         type_nn=self.type_nn, 
                                         size_nn=self.size_nn)        
        self.output_rvpi = Output_rvpi(num_actions=num_actions, 
                                       input_shape=(inplanes, obs_shape[1]//16, obs_shape[1]//16), 
                                       value_prefix=flags.value_prefix,
                                       max_unroll_length=flags.model_k_step_return, 
                                       reward_transform=self.reward_transform, 
                                       zero_init=flags.model_zero_init, 
                                       type_nn=self.type_nn, 
                                       size_nn=self.size_nn)
        if self.reward_transform:
            self.reward_tran = self.output_rvpi.reward_tran

        self.supervise = flags.model_supervise
        self.model_supervise_type = flags.model_supervise_type
        if self.supervise:                        
            if self.model_supervise_type == 0:
                flatten_in_dim = (obs_shape[1]//16)*(obs_shape[2])//16*inplanes          
                self.P_1 = torch.nn.Sequential(nn.Linear(flatten_in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                            nn.Linear(512, 1024), nn.BatchNorm1d(1024))
                self.P_2 = torch.nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1024))
            self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def supervise_loss(self, encodes, x, actions, is_weights, mask, one_hot=False):
        """
        Args:
            encodes(tensor): encodes output from forward with shape (k, B, C, H, W) in the form of z_{t+1}, ..., z_{t+k}
            x(tensor): frames (uint8) with shape (k, B, C, H, W), in the form of s_{t+1}, ..., s{t+k}
            actions(tensor): action (int64) with shape (k, B), in the form of a_{t}, a_{t}, a_{t+1}, .. a_{t+k}
            mask(tensor): mask (float) with shape (k, B)
            im_weights(tensor): importance weight with shape (B) for each sample;  
        Return:
            loss(tensor): scalar self-supervised loss
        """
        k, b, c, h, w = encodes.shape

        encodes = torch.flatten(encodes, 0, 1)        
        encodes = torch.flatten(encodes, 1)
        if self.model_supervise_type == 0:
            src = self.P_2(self.P_1(encodes))
        elif self.model_supervise_type == 1:
            src = encodes
        
        with torch.no_grad():
            x = torch.flatten(x, 0, 1)
            actions = torch.flatten(actions, 0, 1)   
            tgt_encodes = self.encoded(x, actions, one_hot)
            tgt_encodes = torch.flatten(tgt_encodes, 1)
            if self.model_supervise_type == 0:
                tgt = self.P_1(tgt_encodes)
            elif self.model_supervise_type == 1:
                tgt = tgt_encodes
        
        loss = -self.cos(src, tgt.detach())
        loss = loss.view(k, b)
        if mask is not None: loss = loss * mask
        loss = torch.sum(loss, dim=0)
        loss = loss * is_weights
        return torch.sum(loss)
    
    def encoded(self, x, actions, one_hot=False):
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)                
        encodes = self.frameEncoder(x, actions)
        return encodes
        
    def forward(self, x, actions, one_hot=False):
        """
        Args:
            x(tensor): frames (uint8) with shape (B, C, H, W), in the form of s_t
            actions(tensor): action (int64) with shape (k+1, B), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
        Return:
            reward(tensor): predicted reward with shape (k, B, ...), in the form of r_{t+1}, r_{t+2}, ..., r_{t+k}
            value(tensor): predicted value with shape (k+1, B, ...), in the form of v_{t}, v_{t+1}, v_{t+2}, ..., v_{t+k}
            policy(tensor): predicted policy with shape (k+1, B, ...), in the form of pi_{t}, pi_{t+1}, pi_{t+2}, ..., pi_{t+k}
            encoded(tensor): encoded states with shape (k+1, B, ...), in the form of z_t, z_{t+1}, z_{t+2}, ..., z_{t+k}
                Recall we use the transition notation: s_t, a_t, r_{t+1}, s_{t+1}, ...
        """
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)                
        encoded = self.frameEncoder(x, actions[0])
        state = self.output_rvpi.init_state(bsz=x.shape[0], device=x.device)
        return self.forward_encoded(encoded, state, actions[1:], one_hot=True)
    
    def forward_encoded(self, encoded, state, actions, one_hot=False):
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)                
        
        _, _, _, v, v_enc_logits, logits, _ = self.output_rvpi(encoded, predict_reward=False, state=())
        single_r_list, r_list, v_list, logits_list = [], [], [v.unsqueeze(0)], [logits.unsqueeze(0)]
        if self.reward_transform: 
            r_enc_logits_list = []
            v_enc_logits_list = [v_enc_logits.unsqueeze(0)]

        encoded_list = [encoded.unsqueeze(0)]
        
        for k in range(actions.shape[0]):            
            encoded = self.dynamicModel(encoded, actions[k])
            sinlge_r, r, r_enc_logits, v, v_enc_logits, logits, state = self.output_rvpi(encoded, predict_reward=True, state=state)
            if sinlge_r is not None:  single_r_list.append(sinlge_r.unsqueeze(0))
            r_list.append(r.unsqueeze(0))
            v_list.append(v.unsqueeze(0))
            logits_list.append(logits.unsqueeze(0))
            encoded_list.append(encoded.unsqueeze(0))      
            if self.reward_transform: 
                r_enc_logits_list.append(r_enc_logits.unsqueeze(0))
                v_enc_logits_list.append(v_enc_logits.unsqueeze(0))        
        
        rs = torch.concat(r_list, dim=0) if len(r_list) > 0 else None            
        single_rs = torch.concat(single_r_list, dim=0) if len(single_r_list) > 0 else None   
        vs = torch.concat(v_list, dim=0)
        logits = torch.concat(logits_list, dim=0)
        encodes = torch.concat(encoded_list, dim=0)        

        if self.reward_transform:            
            r_enc_logits = torch.concat(r_enc_logits_list, dim=0) if len(r_enc_logits_list) > 0 else None
            v_enc_logits = torch.concat(v_enc_logits_list, dim=0)
        else:
            r_enc_logits = None
            v_enc_logits = None

        #print("actions: ", actions)
        #print("forward encoded out rs: ", rs)
        
        return ModelNetOut(rs=rs, 
                           r_enc_logits=r_enc_logits, 
                           vs=vs, 
                           v_enc_logits=v_enc_logits, 
                           logits=logits, 
                           encodes=encodes, 
                           state=state,
                           single_rs=single_rs)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        device = next(self.parameters()).device
        if device != torch.device("cpu"):
            weights = {k: v.to(device) for k, v in weights.items()}
        self.load_state_dict(weights)        
         
def ModelNet(obs_shape, num_actions, flags):
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