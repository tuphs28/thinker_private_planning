from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from thinker.core.rnn import ConvAttnLSTM
from thinker.net import RewardTran, FrameEncoder, ActorOut

class LegacyActorNet(nn.Module):    
    def __init__(self, obs_shape, gym_obs_shape, num_actions, flags):

        super(LegacyActorNet, self).__init__()
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
        self.model_size_nn = flags.model_size_nn
        
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
                if self.model_type_nn in [2, 3] and self.actor_see_encode:
                    in_channels=64 if self.model_type_nn == 2 else 128
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