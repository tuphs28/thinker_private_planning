from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from thinker.core.rnn import ConvAttnLSTM

ActorOut = namedtuple('ActorOut', ['policy_logits', 'im_policy_logits', 'reset_policy_logits', 
    'action', 'im_action', 'reset_action', 'baseline', 'baseline_enc_s', 'reg_loss'])

def add_hw(x, h, w):
    return x.unsqueeze(-1).unsqueeze(-1).broadcast_to(x.shape + (h,w))

def mlp(input_size, layer_sizes, output_size, output_activation=nn.Identity, activation=nn.ReLU,
    momentum=0.1, init_zero=False):
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
    if init_zero:
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

class ActorNet(nn.Module):    
    def __init__(self, obs_shape, gym_obs_shape, num_actions, flags):

        super(ActorNet, self).__init__()
        self.obs_shape = obs_shape
        self.gym_obs_shape = gym_obs_shape
        self.num_actions = num_actions  
        
        self.tran_t = flags.tran_t                   # number of recurrence of RNN        
        self.tran_mem_n = flags.tran_mem_n           # size of memory for the attn modules
        self.tran_layer_n = flags.tran_layer_n       # number of layers
        self.tran_lstm_no_attn = flags.tran_lstm_no_attn  # to use attention in lstm or not
        self.attn_mask_b = flags.tran_attn_b         # atention bias for current position
        self.conv_out = flags.tran_dim               # size of transformer / LSTM embedding dim        
        self.num_rewards = 2 if (flags.reward_type == 1) else 1 # dim of rewards (1 for vanilla; 2 for planning rewards)
        self.actor_see_p = flags.actor_see_p         # probability of allowing actor to see state
        self.actor_see_encode = flags.actor_see_encode # Whether the actor see the model encoded state or the raw env state
        self.actor_see_double_encode = flags.actor_see_double_encode # Whether the actor see the model encoded state or the raw env state
        self.actor_drc = flags.actor_drc             # Whether to use drc in encoding state
        self.rnn_grad_scale = flags.rnn_grad_scale   # Grad scale for hidden state in RNN
        self.reward_transform = flags.reward_transform # Whether to use reward transform as in MuZero
        self.model_type_nn = flags.model_type_nn
        
        self.conv_out_hw = 1   
        self.d_model = self.conv_out
        
        self.conv1 = nn.Conv2d(in_channels=self.obs_shape[0], out_channels=self.conv_out//2, kernel_size=1, stride=1)        
        self.conv2 = nn.Conv2d(in_channels=self.conv_out//2, out_channels=self.conv_out, kernel_size=1, stride=1)        
        self.frame_conv = torch.nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU())
        self.env_input_size = self.conv_out
        d_in = self.env_input_size + self.d_model 

        self.core = ConvAttnLSTM(h=self.conv_out_hw, w=self.conv_out_hw,
                                input_dim=d_in-self.d_model, hidden_dim=self.d_model,
                                kernel_size=1, num_layers=self.tran_layer_n,
                                num_heads=8, mem_n=self.tran_mem_n, attn=not self.tran_lstm_no_attn,
                                attn_mask_b=self.attn_mask_b, grad_scale=self.rnn_grad_scale)                         
        
        rnn_out_size = self.conv_out_hw * self.conv_out_hw * self.d_model                
        self.fc = nn.Linear(rnn_out_size, 256)        

        last_out_size = 256
        if self.actor_see_p > 0: 
            down_scale_c = 2 if self.actor_see_encode else 4
            last_out_size = last_out_size + (256 if self.actor_drc else (
                256//down_scale_c//4)*(gym_obs_shape[1]//16)*(gym_obs_shape[2]//16))
        self.im_policy = nn.Linear(last_out_size, self.num_actions)        
        self.policy = nn.Linear(last_out_size, self.num_actions)       
        self.baseline = nn.Linear(last_out_size, self.num_rewards)        

        if self.reward_transform:
            self.reward_tran = RewardTran(vec=False)

        self.reset = nn.Linear(last_out_size, 2)        
        
        if self.actor_see_p > 0:
            if not self.actor_drc:
                if not self.actor_see_encode:
                    self.gym_frame_encoder = FrameEncoder(frame_channels=gym_obs_shape[0], num_actions=self.num_actions, 
                        down_scale_c=down_scale_c, concat_action=False)
                if self.model_type_nn == 2 and self.actor_see_encode:
                    in_channels=64
                else:
                    in_channels=256//down_scale_c
                if self.actor_see_double_encode: in_channels=in_channels*2 
                self.gym_frame_conv = torch.nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=256//down_scale_c//2, kernel_size=3, padding='same'), 
                    nn.ReLU(), 
                    nn.Conv2d(in_channels=256//down_scale_c//2, out_channels=256//down_scale_c//4, kernel_size=3, padding='same'), 
                    nn.ReLU())
            else:
                assert not self.actor_see_encode, "actor_drc is not compatiable with actor_see_encode"
                self.gym_frame_conv = torch.nn.Sequential(
                    nn.Conv2d(in_channels=gym_obs_shape[0], out_channels=32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2),
                    nn.ReLU())
                compute_hw_out = lambda hw_in, kernel_size, stride: (hw_in - (kernel_size-1) - 1) // stride + 1
                hw_out_1 = compute_hw_out(gym_obs_shape[1], 8, 4)
                hw_out_2 = compute_hw_out(hw_out_1, 4, 2)
                self.conv_out_hw_2 = hw_out_2
                self.drc_core = ConvAttnLSTM(h=hw_out_2, w=hw_out_2,
                        input_dim=32, hidden_dim=32, kernel_size=3, num_layers=3, 
                        num_heads=8, mem_n=0, attn=False, attn_mask_b=0.,
                        grad_scale=self.rnn_grad_scale)
                self.drc_fc = nn.Linear(hw_out_2*hw_out_2*32, 256)     

        #print("actor size: ", sum(p.numel() for p in self.parameters()))
        #for k, v in self.named_parameters(): print(k, v.numel())   

        self.initial_state(1) # just for setting core_state_sep_ind

    def initial_state(self, batch_size, device=None):
        state = self.core.init_state(batch_size, device=device) 
        self.core_state_sep_ind = len(state)
        if self.actor_see_p > 0 and self.actor_drc:
            state = state + self.drc_core.init_state(batch_size, device=device)
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
                
        x = obs.model_out.unsqueeze(-1).unsqueeze(-1)
        done = obs.done
        
        if len(x.shape) == 4: x = x.unsqueeze(0)
        if len(done.shape) == 1: done = done.unsqueeze(0)  
            
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.  
        env_input = self.frame_conv(x)                
        core_input = env_input.view(T, B, -1, self.conv_out_hw, self.conv_out_hw)
        core_output_list = []
        core_state_1 = core_state[:self.core_state_sep_ind]
        notdone = ~(done.bool())
        for n, (input, nd) in enumerate(zip(core_input.unbind(), notdone.unbind())):  
            # Input shape: B, self.conv_out + self.num_actions + 1, H, W
            for t in range(self.tran_t):                
                if t > 0: nd = torch.ones(B).to(x.device).bool()                    
                nd = nd.view(-1)      
                output, core_state_1 = self.core(input, core_state_1, nd, nd) # output shape: 1, B, core_output_size 
                
            last_input = input   
            core_output_list.append(output)                             
        core_output = torch.cat(core_output_list)              
        core_output = torch.flatten(core_output, 0, 1)        
        core_output = torch.flatten(core_output, start_dim=1)
        core_output = F.relu(self.fc(core_output))

        if self.actor_see_p > 0:
            if not self.actor_see_encode:
                gym_x = obs.gym_env_out.float()
                gym_x = gym_x * obs.see_mask.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                gym_x = torch.flatten(gym_x, 0, 1)   
            if not self.actor_drc:
                if not self.actor_see_encode:
                    conv_out = self.gym_frame_encoder(gym_x, actions = None)   
                else:
                    conv_out = torch.flatten(obs.model_encodes, 0, 1)   
                conv_out = self.gym_frame_conv(conv_out)
                conv_out = torch.flatten(conv_out, start_dim=1)
                core_output = torch.concat([core_output, conv_out], dim=1)
            else:       
                gym_x = gym_x / 255.0
                conv_out = self.gym_frame_conv(gym_x)
                core_input = conv_out.view(T, B, -1, self.conv_out_hw_2, self.conv_out_hw_2)
                core_output_list = []
                core_state_2 = core_state[self.core_state_sep_ind:]
                for n, (input, nd) in enumerate(zip(core_input.unbind(), notdone.unbind())):
                    for t in range(3):                
                        if t > 0: nd = torch.ones(B).to(x.device).bool()                    
                        nd = nd.view(-1)      
                        output, core_state_2 = self.drc_core(input, core_state_2, nd, nd) # output shape: 1, B, core_output_size                         
                    core_output_list.append(output)              

                core_output_2 = torch.cat(core_output_list)   
                core_output_2 = torch.flatten(core_output_2, 0, 1)        
                core_output_2 = torch.flatten(core_output_2, start_dim=1)
                core_output_2 = F.relu(self.drc_fc(core_output_2))
                core_output = torch.concat([core_output, core_output_2], dim=1)
   
        policy_logits = self.policy(core_output)
        im_policy_logits = self.im_policy(core_output)
        reset_policy_logits = self.reset(core_output)

        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        im_action = torch.multinomial(F.softmax(im_policy_logits, dim=1), num_samples=1)
        reset_action = torch.multinomial(F.softmax(reset_policy_logits, dim=1), num_samples=1)
        
        if not self.reward_transform:
            baseline = self.baseline(core_output)
        else:            
            baseline_enc_s = self.baseline(core_output)
            baseline = self.reward_tran.decode(baseline_enc_s)
                   
        reg_loss = (1e-3 * torch.sum(policy_logits**2, dim=-1) / 2 + 
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
            core_state = core_state_1 + core_state_2
        else:
            core_state = core_state_1
        return actor_out, core_state

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

# Model Network

DOWNSCALE_C = 2
   
class FrameEncoder(nn.Module):    
    def __init__(self, num_actions, frame_channels=3, type_nn=0, down_scale_c=2, concat_action=True):
        super(FrameEncoder, self).__init__() 
        self.num_actions = num_actions
        self.type_nn = type_nn
        self.down_scale_c=down_scale_c
        if type_nn in [0, 1]:
            self.concat_action=concat_action
        elif type_nn == 2:
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

        elif type_nn == 2:
            # efficient zero
            out_channels = 64
            self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels // 2)
            self.resblocks1 = nn.ModuleList(
                [ResBlock(out_channels // 2, out_channels // 2) for _ in range(1)]
            )
            self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False,)
            self.downsample_block = ResBlock(out_channels // 2, out_channels, stride=2, downsample=self.conv2)
            self.resblocks2 = nn.ModuleList(
                [ResBlock(out_channels, out_channels) for _ in range(1)]
            )
            self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.resblocks3 = nn.ModuleList(
                [ResBlock(out_channels, out_channels) for _ in range(1)]
            )
            self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)   
            self.resblocks4 = nn.ModuleList(
                [ResBlock(out_channels, out_channels) for _ in range(1)]
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
        elif self.type_nn == 2:
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
    def __init__(self, num_actions, inplanes=256, type_nn=0):        
        super(DynamicModel, self).__init__()
        self.type_nn = type_nn
        if type_nn == 0:
            res = nn.ModuleList([ResBlock(inplanes=inplanes+num_actions, outplanes=inplanes)] + [
                    ResBlock(inplanes=inplanes) for i in range(4)]) 
            self.res = torch.nn.Sequential(*res)
        elif type_nn == 1:                      
            res = nn.ModuleList([ResBlock(inplanes=inplanes) for i in range(15)] + [
                    ResBlock(inplanes=inplanes, outplanes=inplanes*num_actions)])                    
        elif type_nn == 2:
            self.conv1 =  conv3x3(inplanes+num_actions, inplanes)
            self.bn1 = nn.BatchNorm2d(inplanes)
            res = nn.ModuleList([ResBlock(inplanes=inplanes) for i in range(1)])

        self.res = torch.nn.Sequential(*res)
        self.num_actions = num_actions
    
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
        elif self.type_nn == 2:  
            actions = actions.unsqueeze(-1).unsqueeze(-1).tile([1, 1, x.shape[2], x.shape[3]])        
            out = torch.concat([x, actions], dim=1)
            out = self.conv1(out)
            out = self.bn1(out)
            out = out + x
            out = nn.functional.relu(out)
            out = self.res(out)
        return out
    
class Output_rvpi(nn.Module):   
    def __init__(self, num_actions, input_shape, reward_transform, zero_init, type_nn):         
        super(Output_rvpi, self).__init__()        
        c, h, w = input_shape
        self.input_shape = input_shape
        self.type_nn = type_nn
        self.reward_transform = reward_transform

        if self.type_nn in [0, 1]:
            self.conv1 = nn.Conv2d(in_channels=c, out_channels=c//2, kernel_size=3, padding='same') 
            self.conv2 = nn.Conv2d(in_channels=c//2, out_channels=c//4, kernel_size=3, padding='same') 
            fc_in = h * w * (c // 4)        
            self.fc_logits = nn.Linear(fc_in, num_actions)         

            if self.reward_transform:
                self.reward_tran = RewardTran(vec=True)
                self.fc_v = nn.Linear(fc_in, self.reward_tran.encoded_n) 
                self.fc_r = nn.Linear(fc_in, self.reward_tran.encoded_n)         
            else:
                self.fc_v = nn.Linear(fc_in, 1) 
                self.fc_r = nn.Linear(fc_in, 1)         

            if zero_init:
                torch.nn.init.constant_(self.fc_v.weight, 0.)
                torch.nn.init.constant_(self.fc_v.bias, 0.)
                torch.nn.init.constant_(self.fc_r.weight, 0.)
                torch.nn.init.constant_(self.fc_r.bias, 0.)
                torch.nn.init.constant_(self.fc_logits.weight, 0.)
                torch.nn.init.constant_(self.fc_logits.bias, 0.)
        elif self.type_nn == 2:
            self.resblocks = nn.ModuleList(
                [ResBlock(c, c) for _ in range(1)]
            )
            self.conv1x1_v = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=1)
            self.conv1x1_r = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=1)
            self.conv1x1_logits = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=1)
            self.bn_v = nn.BatchNorm2d(16)
            self.bn_r = nn.BatchNorm2d(16)
            self.bn_logits = nn.BatchNorm2d(16)
            if self.reward_transform:
                self.reward_tran = RewardTran(vec=True)
                out_n = self.reward_tran.encoded_n
            else:
                out_n = 1            
            self.fc_v = mlp(h * w * 16, [32], out_n, init_zero=zero_init)
            self.fc_r = mlp(h * w * 16, [32], out_n, init_zero=zero_init)
            self.fc_logits = mlp(h * w * 16, [32], num_actions, init_zero=zero_init)                                  
        
    def forward(self, x):    
        if self.type_nn in [0, 1]:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = torch.flatten(x, start_dim=1)
            x_v, x_r, x_logits = x, x, x
        elif self.type_nn == 2:
            for block in self.resblocks: x = block(x)
            x_v = torch.flatten(nn.functional.relu(self.bn_v(self.conv1x1_v(x))), start_dim=1)
            x_r = torch.flatten(nn.functional.relu(self.bn_r(self.conv1x1_r(x))), start_dim=1)
            x_logits = torch.flatten(nn.functional.relu(self.bn_logits(self.conv1x1_logits(x))), start_dim=1)

        logits = self.fc_logits(x_logits)
        if self.reward_transform:
            v_enc_logits = self.fc_v(x_v)
            v_enc_v = F.softmax(v_enc_logits, dim=-1)
            v_enc_s, v = self.reward_tran.decode(v_enc_v)

            r_enc_logits = self.fc_r(x_r)
            r_enc_v = F.softmax(r_enc_logits, dim=-1)
            r_enc_s, r = self.reward_tran.decode(r_enc_v)
        else:
            v_enc_logits = None
            v = self.fc_v(x_v).squeeze(-1)
            r_enc_logits = None
            r = self.fc_r(x_r).squeeze(-1)
        return r, r_enc_logits, v, v_enc_logits, logits

class ModelNetBase(nn.Module):    
    def __init__(self, obs_shape, num_actions, flags):        
        super(ModelNetBase, self).__init__()      
        self.rnn = False
        self.flags = flags        
        self.obs_shape = obs_shape
        self.num_actions = num_actions          
        self.reward_transform = flags.reward_transform
        self.type_nn = flags.model_type_nn # type_nn: type of neural network for the model; 0 for small, 1 for large
        self.frameEncoder = FrameEncoder(num_actions=num_actions, frame_channels=obs_shape[0], type_nn=self.type_nn)
        if self.type_nn in [0, 1]:
            inplanes = 256 // DOWNSCALE_C
        else:
            inplanes = 64
        self.dynamicModel = DynamicModel(num_actions=num_actions, inplanes=inplanes, type_nn=self.type_nn)
        self.output_rvpi = Output_rvpi(num_actions=num_actions, input_shape=(inplanes, 
                      obs_shape[1]//16, obs_shape[1]//16), reward_transform=self.reward_transform, 
                      zero_init=flags.model_zero_init, type_nn=self.type_nn)
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

    def supervise_loss(self, encodeds, x, actions, is_weights, one_hot=False):
        """
        Args:
            encodes(tensor): encodes output from forward with shape (k, B, C, H, W) in the form of z_{t+1}, ..., z_{t+k}
            x(tensor): frames (uint8) with shape (k, B, C, H, W), in the form of s_{t+1}, ..., s{t+k}
            actions(tensor): action (int64) with shape (k, B), in the form of a_{t}, a_{t}, a_{t+1}, .. a_{t+k}
            im_weights(tensor): importance weight with shape (B) for each sample;  
        Return:
            loss(tensor): scalar self-supervised loss
        """
        k, bsz, c, h, w = encodeds.shape

        encodeds = torch.flatten(encodeds, 0, 1)        
        encodeds = torch.flatten(encodeds, 1)
        if self.model_supervise_type == 0:
            src = self.P_2(self.P_1(encodeds))
        elif self.model_supervise_type == 1:
            src = encodeds
        
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
        loss = torch.sum(loss.reshape(k, bsz), dim=0)
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
            reward(tensor): predicted reward with shape (k, B), in the form of r_{t+1}, r_{t+2}, ..., r_{t+k}
            value(tensor): predicted value with shape (k+1, B), in the form of v_{t}, v_{t+1}, v_{t+2}, ..., v_{t+k}
            policy(tensor): predicted policy with shape (k+1, B), in the form of pi_{t}, pi_{t+1}, pi_{t+2}, ..., pi_{t+k}
            encoded(tensor): encoded states with shape (k+1, B), in the form of z_t, z_{t+1}, z_{t+2}, ..., z_{t+k}
                Recall we use the transition notation: s_t, a_t, r_{t+1}, s_{t+1}, ...
        """
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)                
        encoded = self.frameEncoder(x, actions[0])
        return self.forward_encoded(encoded, actions[1:], one_hot=True)
    
    def forward_encoded(self, encoded, actions, one_hot=False):
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)                
        
        r, r_enc_logits, v, v_enc_logits, logits = self.output_rvpi(encoded)
        r_list, v_list, logits_list = [], [v.unsqueeze(0)], [logits.unsqueeze(0)]
        if self.reward_transform: 
            r_enc_logits_list = []
            v_enc_logits_list = [v_enc_logits.unsqueeze(0)]

        encoded_list = [encoded.unsqueeze(0)]
        
        for k in range(actions.shape[0]):            
            encoded = self.dynamicModel(encoded, actions[k])
            r, r_enc_logits, v, v_enc_logits, logits = self.output_rvpi(encoded)
            r_list.append(r.unsqueeze(0))
            v_list.append(v.unsqueeze(0))
            logits_list.append(logits.unsqueeze(0))
            encoded_list.append(encoded.unsqueeze(0))      
            if self.reward_transform: 
                r_enc_logits_list.append(r_enc_logits.unsqueeze(0))
                v_enc_logits_list.append(v_enc_logits.unsqueeze(0))
        
        
        rs = torch.concat(r_list, dim=0) if len(r_list) > 0 else None            
        vs = torch.concat(v_list, dim=0)
        logits = torch.concat(logits_list, dim=0)
        encodeds = torch.concat(encoded_list, dim=0)        

        if self.reward_transform:            
            r_enc_logits = torch.concat(r_enc_logits_list, dim=0) if len(r_enc_logits_list) > 0 else None
            v_enc_logits = torch.concat(v_enc_logits_list, dim=0)
        else:
            r_enc_logits = None
            v_enc_logits = None
        
        return rs, r_enc_logits, vs, v_enc_logits, logits, encodeds        

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        device = next(self.parameters()).device
        if device != torch.device("cpu"):
            weights = {k: v.to(device) for k, v in weights.items()}
        self.load_state_dict(weights)        


class ModelNetRNN(nn.Module):    
    def __init__(self, obs_shape, num_actions, flags):        
        super(ModelNetRNN, self).__init__()      
        self.rnn = True
        self.flags = flags
        self.obs_shape = obs_shape
        self.num_actions = num_actions 

        self.tran_t = 1
        self.tran_mem_n = 0
        self.tran_layer_n = 1
        self.tran_lstm_no_attn = True
        self.attn_mask_b = 0                
        
        self.conv_out = 32        
        self.conv_out_hw = 8
        #self.conv1 = nn.Conv2d(in_channels=self.obs_shape[0], out_channels=self.conv_out, kernel_size=8, stride=4)        
        #self.conv2 = nn.Conv2d(in_channels=self.conv_out, out_channels=self.conv_out, kernel_size=4, stride=2)                
        #self.frame_conv = torch.nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU())

        self.conv_out = 32
        self.conv_out_hw = 5        
        self.frameEncoder = FrameEncoder(num_actions=self.num_actions)
        self.frame_conv = torch.nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128//2, kernel_size=3, padding='same'), 
                nn.ReLU(), 
                nn.Conv2d(in_channels=128//2, out_channels=128//4, kernel_size=3, padding='same'), 
                nn.ReLU())

        self.debug = flags.model_rnn_debug
        self.disable_mem = flags.model_disable_mem

        if self.debug:
            self.policy = nn.Linear(5*5*32, self.num_actions)        
            self.baseline = nn.Linear(5*5*32, 1)      
            self.r = nn.Linear(5*5*32, 1)    
        else:
            self.env_input_size = self.conv_out 
            self.d_model = self.conv_out 
            d_in = self.env_input_size + self.d_model 

            self.core = ConvAttnLSTM(h=self.conv_out_hw, w=self.conv_out_hw,
                                    input_dim=d_in-self.d_model, hidden_dim=self.d_model,
                                    kernel_size=3, num_layers=self.tran_layer_n,
                                    num_heads=8, mem_n=self.tran_mem_n, attn=not self.tran_lstm_no_attn,
                                    attn_mask_b=self.attn_mask_b, grad_scale=1)                         
            
            rnn_out_size = self.conv_out_hw * self.conv_out_hw * self.d_model                
            self.fc = nn.Linear(rnn_out_size, 256)        
                
            self.policy = nn.Linear(256, self.num_actions)        
            self.baseline = nn.Linear(256, 1)        
            

    def init_state(self, bsz, device=None):
        if self.debug:
            return (torch.zeros(1, bsz, 1, 1, 1),)
        return self.core.init_state(bsz, device)
        
    def forward(self, x, actions, done, state, one_hot=False):
        """
        Args:
            x(tensor): frames (uint8 or float) with shape (T, B, C, H, W), in the form of s_t
            actions(tensor): action (int64) with shape (T, B) or (T, B, num_actions)
            done(tensor): done (bool) with shape (T, B)
            state(tuple): tuple of inital state
            one_hot(bool): whether the actions are in one-hot encoding
        Return:
            vs(tensor): values (float) with shape (T, B)
            logits(tensor): policy logits (float) with shape (T, B, num_actions)
            state(tuple): tuple of state tensor after the last step
        """
        assert done.dtype == torch.bool, "done has to be boolean"       

        T, B = x.shape[0], x.shape[1]

        if one_hot: 
            assert actions.shape == (T, B, self.num_actions), ("invalid action shape:", actions.shape)
        else:
            assert actions.shape == (T, B,),  ("invalid action shape:", actions.shape)
        assert len(x.shape) == 5

        #x = x.float() / 255.0  
        x = torch.flatten(x, 0, 1)        
        if not one_hot:
            actions = F.one_hot(actions.view(T * B), self.num_actions).float()
        else:
            actions = actions.view(T * B, -1)
        
        #conv_out = self.frame_conv(x)   
        conv_out = self.frameEncoder(x, actions)   
        conv_out = self.frame_conv(conv_out) 

        if self.debug:
            core_output = torch.flatten(conv_out, start_dim=1)
            vs = self.baseline(core_output).view(T, B)
            logits = self.policy(core_output).view(T, B, self.num_actions)
            state = self.init_state(B, x.device)
            return vs, logits, state

        core_input = conv_out
        core_input = core_input.view(T, B, self.env_input_size, self.conv_out_hw, self.conv_out_hw)
        core_output_list = []

        if self.disable_mem:
            state = self.init_state(bsz=B, device=x.device)
        notdone = (~done).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            for t in range(self.tran_t):                          
                nd_ = nd if t == 0 else torch.ones_like(nd)
                output, state = self.core(input, state, nd_, nd_) # output shape: 1, B, core_output_size 
            core_output_list.append(output)    
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        core_output = F.relu(self.fc(torch.flatten(core_output, start_dim=1)))                   
        vs = self.baseline(core_output).view(T, B)
        logits = self.policy(core_output).view(T, B, self.num_actions)
        return vs, logits, state

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
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