from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from thinker import util
from thinker.core.rnn import ConvAttnLSTM
from thinker.core.module import MLP
from thinker.model_net import BaseNet, RVTran

ActorOut = namedtuple(
    "ActorOut",
    [     
        "pri", # sampled primiary action
        "reset", # sampled reset action
        "action", # tuple of the above two actions 
        "action_prob", # prob of primary action 
        "c_action_log_prob", # log prob of chosen action
        "baseline", # baseline 
        "baseline_enc", # baseline encoding, only for non-scalar enc_type
        "entropy_loss", # entropy loss
        "reg_loss", # regularization loss
        "misc",
    ],
)

def compute_action_log_prob(logits, actions):
    return -torch.nn.CrossEntropyLoss(reduction="none")(
        input=torch.flatten(logits, 0, 1), target=torch.flatten(actions, 0, 1)
    ).view_as(actions)

def add_hw(x, h, w):
    return x.unsqueeze(-1).unsqueeze(-1).broadcast_to(x.shape + (h, w))

class ShallowThreeDEncoder(nn.Module):
    # shallow processor for 3d inputs; can be applied to model's hidden state or predicted real state
    def __init__(self, 
                 input_shape, 
                 out_size=256):
        super(ShallowThreeDEncoder, self).__init__()
        self.input_shape = input_shape
        self.out_size = out_size

        c, h, w = self.input_shape
        compute_hw = lambda h, w, k, s: ((h - k) // s + 1,  (h - k) // s + 1)
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        h, w = compute_hw(h, w, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        h, w = compute_hw(h, w, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        h, w = compute_hw(h, w, 3, 1)
        fc_in_size = h * w * 64
        # Fully connected layer.
        self.fc = nn.Linear(fc_in_size, out_size)

    def forward(self, x):
        """encode the state or model's encoding inside the actor network
        args:
            x: input tensor of shape (B, C, H, W); can be state or model's encoding
        return:
            output: output tensor of shape (B, self.out_size)"""

        assert x.dtype in [torch.float, torch.float16]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class ThreeDEncoder(nn.Module):
    # processor for 3d inputs; can be applied to model's hidden state or predicted real state
    def __init__(self, 
                 input_shape, 
                 downpool=False, 
                 firstpool=False,    
                 out_size=256,
                 see_double=False):
        super(ThreeDEncoder, self).__init__()
        if see_double:
            input_shape = (input_shape[0] // 2,) + tuple(input_shape[1:])
        self.input_shape = input_shape        
        self.downpool = downpool
        self.firstpool = firstpool
        self.out_size = out_size
        self.see_double = see_double        

        # following code is from Torchbeast, which is the same as Impala deep model
        in_channels = input_shape[0]
        conv_out_h = input_shape[1]
        conv_out_w = input_shape[2]

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []

        if firstpool:
            self.down_pool_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=16,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            in_channels = 16
            conv_out_h = (conv_out_h - 1) // 2 + 1
            conv_out_w = (conv_out_w - 1) // 2 + 1

        num_chs = [16, 32, 32] if downpool else [64, 64, 32]
        for num_ch in num_chs:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            if downpool:
                feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                conv_out_h = (conv_out_h - 1) // 2 + 1
                conv_out_w = (conv_out_w - 1) // 2 + 1
            self.feat_convs.append(nn.Sequential(*feats_convs))
            in_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        # out shape after conv is: (num_ch, input_shape[1], input_shape[2])
        core_out_size = num_ch * conv_out_h * conv_out_w
        mlp_out_size = self.out_size if not self.see_double else self.out_size // 2
        self.fc = nn.Sequential(nn.Linear(core_out_size, mlp_out_size), nn.ReLU())

    def forward(self, x):
        if not self.see_double:
            return self.forward_single(x)
        else:
            out_1 = self.forward_single(x[:, :self.input_shape[0]])
            out_2 = self.forward_single(x[:, self.input_shape[0]:])            
            return torch.concat([out_1, out_2], dim=1)

    def forward_single(self, x):
        """encode the state or model's encoding inside the actor network
        args:
            x: input tensor of shape (B, C, H, W); can be state or model's encoding
        return:
            output: output tensor of shape (B, self.out_size)"""
        assert x.dtype in [torch.float, torch.float16]
        if self.firstpool:
            x = self.down_pool_conv(x)
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = torch.flatten(x, start_dim=1)
        x = self.fc(F.relu(x))
        return x

class RNNEncoder(nn.Module):
    # RNN processor for 1d inputs; can be used directly on tree rep or encoded 3d input
    def __init__(self, 
                 in_size, # int; input size
                 flags            
                 ):
        super(RNNEncoder, self).__init__()  
        self.rnn_in_fc = nn.Sequential(
                    nn.Linear(in_size, flags.tran_dim), nn.ReLU()
        )  
        self.tran_layer_n = flags.tran_layer_n 
        if self.tran_layer_n > 0:
            self.rnn = ConvAttnLSTM(
                input_dim=flags.tran_dim,
                hidden_dim=flags.tran_dim,
                num_layers=flags.tran_layer_n,
                attn=not flags.tran_lstm_no_attn,
                mem_n=flags.tran_mem_n,
                num_heads=flags.tran_head_n,
                attn_mask_b=flags.tran_attn_b,
                tran_t=flags.tran_t,
            ) 
        self.rnn_out_fc = nn.Sequential(
            nn.Linear(flags.tran_dim, flags.tran_dim), nn.ReLU()
        )

    def initial_state(self, batch_size=1, device=None):
        if self.tran_layer_n > 0:
            return self.rnn.initial_state(batch_size, device=device)
        else:
            return ()

    def forward(self, x, done, core_state):
        # input should have shape (T*B, C) 
        # done should have shape (T, B)
        T, B = done.shape
        x = self.rnn_in_fc(x)
        if self.tran_layer_n >= 1:
            x = x.view(*((T, B) + x.shape[1:])).unsqueeze(-1).unsqueeze(-1)            
            core_output, core_state = self.rnn(x, done, core_state)
            core_output = torch.flatten(core_output, 0, 1)
            d = torch.flatten(core_output, 1)   
        else:
            d = x     
        d = self.rnn_out_fc(d)
        return d, core_state

class ActorNetBase(BaseNet):
    def __init__(self, obs_space, action_space, flags):
        super(ActorNetBase, self).__init__()

        self.disable_thinker = flags.wrapper_type == 1
        self.see_double = flags.return_double
        self.see_tree_rep = flags.see_tree_rep and not self.disable_thinker
        if self.see_tree_rep:
            self.tree_reps_shape = obs_space["tree_reps"].shape[1:]             
        self.see_h = flags.see_h and not self.disable_thinker
        if self.see_h:
            self.hs_shape = obs_space["hs"].shape[1:]
            self.oned_h = len(self.hs_shape) == 1
        self.see_x = flags.see_x
        if self.see_x and not self.disable_thinker:
            self.xs_shape = obs_space["xs"].shape[1:]
            self.oned_x = len(self.xs_shape) == 1
        self.see_real_state = flags.see_real_state        
        if flags.see_real_state:
            self.real_states_shape = obs_space["real_states"].shape[1:]  
            self.register_buffer("norm_low", torch.tensor(obs_space["real_states"].low[0,]))
            self.register_buffer("norm_high", torch.tensor(obs_space["real_states"].high[0,]))     

        if not self.disable_thinker:
            self.num_actions = action_space[0][0].n
        else:
            self.num_actions = action_space[0].n

        self.tran_dim = flags.tran_dim 
        self.tree_rep_rnn = flags.tree_rep_rnn and flags.see_tree_rep 
        self.x_rnn = flags.x_rnn and flags.see_x  

        self.num_rewards = 1
        self.num_rewards += int(flags.im_cost > 0.0)
        self.num_rewards += int(flags.cur_cost > 0.0)

        self.enc_type = flags.critic_enc_type  
        self.critic_zero_init = flags.critic_zero_init        

        self.sep_im_head = flags.sep_im_head
        self.last_layer_n = flags.last_layer_n

        self.float16 = flags.float16
        self.flags = flags        

        aux_info_size = flags.max_depth * self.num_actions + 4 + flags.rec_t

        # encoder for state or encoding output
        last_out_size = self.num_actions + self.num_rewards
        if not self.disable_thinker:
            last_out_size += 2

        if self.see_h:
            if not self.oned_x:
                self.h_encoder = ThreeDEncoder(
                    input_shape=self.hs_shape,                                 
                    see_double=self.see_double
                )
                h_out_size = self.h_encoder.out_size
            else:
                self.h_encoder = MLP(
                    input_size=self.hs_shape[0],
                    layer_sizes=[200, 200, 200],
                    output_size=100,
                    norm=False,
                    skip_connection=True,
                )
                h_out_size = 100
            last_out_size += h_out_size
        
        if self.see_x:
            if not self.oned_x:
                self.x_encoder_pre = ThreeDEncoder(
                    input_shape=self.xs_shape, 
                    downpool=True,
                    firstpool=flags.x_enc_first_pool,
                    see_double=self.see_double
                )
                x_out_size = self.x_encoder_pre.out_size
            else:
                self.x_encoder_pre = MLP(
                    input_size=self.xs_shape[0],
                    layer_sizes=[200, 200, 200],
                    output_size=100,
                    norm=False,
                    skip_connection=True,
                )
                x_out_size = 100

            if self.x_rnn:
                x_rnn_in_size = x_out_size
                x_rnn_in_size += aux_info_size
                self.x_encoder_rnn = RNNEncoder(
                    in_size=x_rnn_in_size,
                    flags=flags,
                )
                x_out_size = flags.tran_dim
            
            last_out_size += x_out_size

        if self.see_real_state:
            self.real_state_encoder =  ThreeDEncoder(
                input_shape=self.xs_shape, 
                num_actions=self.num_actions, 
                downpool=True,
                firstpool=flags.x_enc_first_pool,
                see_double=self.see_double
            )
            last_out_size += self.real_state_encoder.out_size
                    
        if self.see_tree_rep:            
            if self.tree_rep_rnn:
                self.tree_rep_encoder = RNNEncoder(
                    in_size=self.tree_reps_shape[0],
                    flags=flags
                )
                last_out_size += flags.tran_dim
            else:
                self.tree_rep_encoder = MLP(
                    input_size=self.tree_reps_shape[0],
                    layer_sizes=[200, 200, 200],
                    output_size=100,
                    norm=False,
                    skip_connection=True,
                )
                last_out_size += 100        

        if self.last_layer_n > 0:
            self.final_mlp =  MLP(
                input_size=last_out_size,
                layer_sizes=[200]*self.last_layer_n,
                output_size=100,
                norm=False,
                skip_connection=True,
            )
            last_out_size = 100

        self.policy = nn.Linear(last_out_size, self.num_actions)

        if not self.disable_thinker:
            if self.sep_im_head:
                self.im_policy = nn.Linear(last_out_size, self.num_actions)
            self.reset = nn.Linear(last_out_size, 2)

        self.rv_tran = None
        if self.enc_type == 0:
            self.baseline = nn.Linear(last_out_size, self.num_rewards)
            if self.flags.reward_clip > 0:
                self.baseline_clamp = self.flags.reward_clip / (
                    1 - self.flags.discounting
                )
        elif self.enc_type == 1:
            self.baseline = nn.Linear(last_out_size, self.num_rewards)
            self.rv_tran = RVTran(enc_type=self.enc_type, enc_f_type=flags.critic_enc_f_type)
        elif self.enc_type in [2, 3]:                        
            self.rv_tran = RVTran(enc_type=self.enc_type, enc_f_type=flags.critic_enc_f_type)
            self.out_n = self.rv_tran.encoded_n
            self.baseline = nn.Linear(last_out_size, self.num_rewards * self.out_n)            

        if self.critic_zero_init:
            nn.init.constant_(self.baseline.weight, 0.0)
            nn.init.constant_(self.baseline.bias, 0.0)

        self.initial_state(batch_size=1) # initialize self.state_idx


    def initial_state(self, batch_size, device=None):
        self.state_idx = {}
        idx = 0
        initial_state = ()
        if self.x_rnn:
            core_state = self.x_encoder_rnn.initial_state(batch_size, device=device)
            initial_state = initial_state + core_state
            self.state_idx["x"] = slice(idx, idx+len(core_state))
            idx += len(core_state)
        
        if self.tree_rep_rnn:
            core_state = self.tree_rep_encoder.initial_state(batch_size, device=device)
            initial_state = initial_state + core_state
            self.state_idx["tree_rep"] = slice(idx, idx+len(core_state))
            idx += len(core_state)

        self.state_len = idx
        return initial_state
    
    def forward(self, 
                env_out, 
                core_state=(), 
                clamp_action=None, 
                compute_loss=False,
                greedy=False,
                dbg_mode=0,
                ):
        """one-step forward for the actor;
        args:
            env_out (EnvOut):
                tree_reps (tensor): tree_reps output with shape (T x B x C)
                xs (tensor): optional - model predicted state with shape (T x B x C X H X W)                
                hs (tensor): optional - hidden state with shape (T x B x C X H X W)                
                real_states (tensor): optional - root's real state with shape (T x B x C X H X W)                
                done  (tensor): if episode ends with shape (T x B)
                step_status (tensor): current step status with shape (T x B)
                last_pri (tensor): last primiary action (non-one-hot) with shape (T x B)
                last_reset (tensor): last reset action (non-one-hot) with shape (T x B)
                and other environment output that is not used.
            core_state (tuple): rnn state of the actor network
            clamp_action (tuple): option - if not none, the sampled action will be set to this action;
                the main purpose is for computing c_action_log_prob
            compute_loss (boolean): wheather to return entropy loss and reg loss
            greedy (bool): whether to sample greedily
        return:
            ActorOut:
                see definition of ActorOut; this is a tuple with elements of 
                    shape (T x B x ...) except actor_out.action, which is a 
                    tuple of primiary and reset action, each with shape (B,),
                    selected on the last step
        """
        done = env_out.done
        assert (
            len(done.shape) == 2
        ), f"done shape should be (T, B) instead of {done.shape}"
        T, B = done.shape

        assert len(core_state) == self.state_len, "core_state should have length %d" % self.state_len
        new_core_state = [None] * self.state_len

        final_out = []

        last_pri = torch.flatten(env_out.last_pri, 0, 1)
        last_pri = F.one_hot(last_pri, self.num_actions)
        final_out.append(last_pri)

        if not self.disable_thinker:
            last_reset = torch.flatten(env_out.last_reset, 0, 1)
            last_reset = F.one_hot(last_reset, 2)
            final_out.append(last_reset)

        reward = env_out.reward
        reward[torch.isnan(reward)] = 0.
        last_reward = torch.clamp(torch.flatten(reward, 0, 1), -1, +1)
        final_out.append(last_reward)

        if self.see_tree_rep:            
            tree_rep = torch.flatten(env_out.tree_reps, 0, 1)
            if self.tree_rep_rnn:
                core_state_ = core_state[self.state_idx['tree_rep']]
                encoded_tree_rep, core_state_ = self.tree_rep_encoder(
                    tree_rep, done, core_state_)
                new_core_state[self.state_idx['tree_rep']] = core_state_
            else:
                encoded_tree_rep = self.tree_rep_encoder(tree_rep)
            final_out.append(encoded_tree_rep)
        
        if self.see_h:
            hs = torch.flatten(env_out.hs, 0, 1)
            encoded_h = self.h_encoder(hs)
            final_out.append(encoded_h)

        if self.see_x:
            xs = torch.flatten(env_out.xs, 0, 1)
            with autocast(enabled=self.float16):                
                encoded_x = self.x_encoder_pre(xs)
            if self.float16: encoded_x = encoded_x.float()
                
            if self.x_rnn:
                core_state_ = core_state[self.state_idx['x']]
                aux_info = util.mask_tree_rep(tree_rep, self.num_actions, self.flags.rec_t)
                encoded_x = torch.concat([encoded_x, aux_info], dim=1)
                encoded_x, core_state_ = self.x_encoder_rnn(
                    encoded_x, done, core_state_)
                new_core_state[self.state_idx['x']] = core_state_
            
            final_out.append(encoded_x)

        if self.see_real_state:
            real_states = torch.flatten(env_out.real_states, 0, 1)            
            assert real_states.dtype == torch.uint8
            encoded_real_state = (real_states.float() - self.norm_low) / \
                    (self.norm_high -  self.norm_low)
            with autocast(enabled=self.float16):      
                encoded_real_state = self.real_state_encoder(encoded_real_state)
            if self.float16: encoded_real_state = encoded_real_state.float()
            final_out.append(encoded_real_state)

        final_out = torch.concat(final_out, dim=-1)   

        if self.last_layer_n > 0:
            final_out = self.final_mlp(final_out)     

        # compute logits
        pri_logits = self.policy(final_out)    
        if not self.disable_thinker:
            if self.sep_im_head:
                im_logits = self.im_policy(final_out)
                im_mask = env_out.step_status <= 1 # imagainary action will be taken next
                im_mask = torch.flatten(im_mask, 0, 1).unsqueeze(-1)
                pri_logits = torch.where(im_mask, im_logits, pri_logits)
            reset_logits = self.reset(final_out)
        else:   
            reset_logits = None

        # compute entropy loss
        if compute_loss:
            entropy_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                input=pri_logits, target=F.softmax(pri_logits, dim=-1)
            )
            entropy_loss = entropy_loss.view(T, B)
            if not self.disable_thinker:
                ent_reset_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                    input=reset_logits, target=F.softmax(reset_logits, dim=-1)
                )
                ent_reset_loss = ent_reset_loss.view(T, B) * (env_out.step_status <= 1).float()
                entropy_loss = entropy_loss + ent_reset_loss 
        else:
            entropy_loss = None

        # sample action
        pri = self.sample(pri_logits, greedy=greedy, dim=1)
        pri_logits = pri_logits.view(T, B, self.num_actions)
        pri = pri.view(T, B)
        if not self.disable_thinker:
            reset = self.sample(reset_logits, greedy=greedy, dim=1)
            reset_logits = reset_logits.view(T, B, 2)
            reset = reset.view(T, B)    
        else:
            reset = None

        # clamp the action to clamp_action
        if clamp_action is not None:
            if not self.disable_thinker:
                pri[:clamp_action[0].shape[0]] = clamp_action[0]
                reset[:clamp_action[1].shape[0]] = clamp_action[1]
            else:
                pri[:clamp_action.shape[0]] = clamp_action

        # compute chosen log porb
        c_action_log_prob = compute_action_log_prob(pri_logits, pri)     
        if not self.disable_thinker:
            c_reset_log_prob = compute_action_log_prob(reset_logits, reset)     
            c_reset_log_prob = c_reset_log_prob * (env_out.step_status <= 1).float()
            # if next action is real action, reset will never be used
            c_action_log_prob += c_reset_log_prob

        # pack last step's action and action prob
        if not self.disable_thinker:
            action = (pri[-1], reset[-1])
        else:
            action = pri[-1]            
        action_prob = F.softmax(pri_logits, dim=-1)        

        # compute baseline
        if self.enc_type == 0:
            baseline = self.baseline(final_out)
            if self.flags.reward_clip > 0:
                baseline = torch.clamp(
                    baseline, -self.baseline_clamp, +self.baseline_clamp
                )
            baseline_enc = None
        elif self.enc_type == 1:
            baseline_enc_s = self.baseline(final_out)
            baseline = self.rv_tran.decode(baseline_enc_s)
            baseline_enc = baseline_enc_s
        elif self.enc_type in [2, 3]:
            baseline_enc_logit = self.baseline(final_out).reshape(
                T * B, self.num_rewards, self.out_n
            )
            baseline_enc_v = F.softmax(baseline_enc_logit, dim=-1)
            baseline = self.rv_tran.decode(baseline_enc_v)
            baseline_enc = baseline_enc_logit

        baseline_enc = (
            baseline_enc.view((T, B) + baseline_enc.shape[1:])
            if baseline_enc is not None
            else None
        )
        baseline = baseline.view(T, B, self.num_rewards)

        if compute_loss:
            reg_loss = (
                1e-3 * torch.sum(pri_logits**2, dim=-1) / 2
                + 1e-6 * torch.sum(final_out**2, dim=-1).view(T, B) / 2
            )
            if not self.disable_thinker:
                reg_loss += (
                    + 1e-3 * torch.sum(reset_logits**2, dim=-1) / 2
                )
        else:
            reg_loss = None

        misc = {
            "pri_logits": pri_logits,
            "reset_logits": reset_logits,
        }

        actor_out = ActorOut(
            pri=pri,
            reset=reset,
            action=action,
            action_prob=action_prob,
            c_action_log_prob=c_action_log_prob,            
            baseline=baseline,
            baseline_enc=baseline_enc,
            entropy_loss=entropy_loss,
            reg_loss=reg_loss,
            misc=misc,
        )
        core_state = tuple(new_core_state)

        return actor_out, core_state
    
    def sample(self, logits, greedy, dim=-1):
        if not greedy:
            gumbel_noise = torch.empty_like(logits).uniform_().clamp(1e-10, 1).log().neg_().clamp(1e-10, 1).log().neg_()
            sampled_action = (logits + gumbel_noise).argmax(dim=dim)
            return sampled_action.detach()
        else:
            return torch.argmax(logits, dim=dim)

class DRCNet(BaseNet):
    # Deprecated, not yet updated
    def __init__(self, obs_shape, gym_obs_shape, num_actions, flags):
        super(DRCNet, self).__init__()
        assert flags.disable_model
        assert flags.critic_enc_type == 0

        self.obs_shape = obs_shape
        self.gym_obs_shape = gym_obs_shape
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2
            ),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
        )
        output_shape = lambda h, w, kernel, stride, padding: (
            ((h + 2 * padding - kernel) // stride + 1),
            ((w + 2 * padding - kernel) // stride + 1),
        )

        h, w = output_shape(gym_obs_shape[1], gym_obs_shape[2], 8, 4, 2)
        h, w = output_shape(h, w, 4, 2, 1)

        self.drc_depth = 3
        self.drc_n = 3
        self.core = ConvAttnLSTM(
            h=h,
            w=w,
            input_dim=32,
            hidden_dim=32,
            kernel_size=3,
            num_layers=3,
            num_heads=8,
            mem_n=None,
            attn=False,
            attn_mask_b=None,
            pool_inject=True,
        )
        last_out_size = 32 * h * w * 2
        self.final_layer = nn.Linear(last_out_size, 256)
        self.policy = nn.Linear(256, self.num_actions)
        self.baseline = nn.Linear(256, 1)

    def initial_state(self, batch_size, device=None):
        return self.core.initial_state(batch_size, device=device)

    def forward(self, obs, core_state=(), greedy=False):
        done = obs.done
        assert (
            len(done.shape) == 2
        ), f"done shape should be (T, B) instead of {done.shape}"
        T, B = done.shape
        model_enc = obs.gym_env_out.float() / 255.0
        model_enc = torch.flatten(model_enc, 0, 1)
        model_enc = self.encoder(model_enc)
        core_input = model_enc.view(*((T, B) + model_enc.shape[1:]))
        core_output_list = []
        notdone = ~(done.bool())
        for n, (input, nd) in enumerate(zip(core_input.unbind(), notdone.unbind())):
            for t in range(self.drc_n):
                if t > 0:
                    nd = torch.ones_like(nd)
                nd = nd.view(-1)
                output, core_state = self.core(input, core_state, nd, nd)
            core_output_list.append(output)
        core_output = torch.cat(core_output_list)
        core_output = torch.flatten(core_output, 0, 1)
        core_output = torch.cat([model_enc, core_output], dim=1)
        core_output = torch.flatten(core_output, 1)
        final_out = F.relu(self.final_layer(core_output))
        policy_logits = self.policy(final_out)
        if not greedy:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        action = action.view(T, B)
        baseline = self.baseline(final_out).view(T, B, 1)
        reg_loss = (
            1e-3 * torch.sum(policy_logits**2, dim=-1)
            + 1e-5 * torch.sum(torch.square(self.baseline.weight))
            + 1e-5 * torch.sum(torch.square(self.policy.weight))
        )
        actor_out = ActorOut(
            policy_logits=policy_logits,
            reset_policy_logits=None,
            action=action,
            reset=None,
            baseline_enc=None,
            baseline=baseline,
            reg_loss=reg_loss,
        )
        return actor_out, core_state


def ActorNet(*args, **kwargs):
    if kwargs["flags"].drc:
        return DRCNet(*args, **kwargs)
    else:
        return ActorNetBase(*args, **kwargs)
