from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from thinker import util
from thinker.core.rnn import ConvAttnLSTM, LSTMReset
from thinker.core.module import MLP, OneDResBlock
from thinker.model_net import BaseNet, RVTran
from gym import spaces

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
    assert len(logits.shape) == len(actions.shape) + 1
    has_dim = len(actions.shape) == 3    
    end_dim = 2 if has_dim else 1
    log_prob = -torch.nn.CrossEntropyLoss(reduction="none")(
            input=torch.flatten(logits, 0, end_dim), target=torch.flatten(actions, 0, end_dim)
    )
    log_prob = log_prob.view_as(actions)
    if has_dim:
        log_prob = torch.sum(log_prob, dim=-1)
    return log_prob

def add_hw(x, h, w):
    return x.unsqueeze(-1).unsqueeze(-1).broadcast_to(x.shape + (h, w))

class ShallowAFrameEncoder(nn.Module):
    # shallow processor for 3d inputs; can be applied to model's hidden state or predicted real state
    def __init__(self, 
                 input_shape, 
                 out_size=256):
        super(ShallowAFrameEncoder, self).__init__()
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

class AFrameEncoder(nn.Module):
    # processor for 3d inputs; can be applied to model's hidden state or predicted real state
    def __init__(self, 
                 input_shape, 
                 downpool=False, 
                 firstpool=False,    
                 out_size=256,
                 see_double=False):
        super(AFrameEncoder, self).__init__()
        if see_double:
            input_shape = (input_shape[0] // 2,) + tuple(input_shape[1:])
        self.input_shape = input_shape        
        self.downpool = downpool
        self.firstpool = firstpool
        self.out_size = out_size
        self.see_double = see_double        

        self.oned_input = len(self.input_shape) == 1
        in_channels = input_shape[0]
        if not self.oned_input:
            # following code is from Torchbeast, which is the same as Impala deep model            
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
        else:
            n_block = 2 
            hidden_size = 256
            self.hidden_size = hidden_size
            self.input_block = nn.Sequential(
                nn.Linear(in_channels, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Tanh()
            )            
            self.res = nn.Sequential(*[OneDResBlock(hidden_size) for _ in range(n_block)])
            core_out_size = hidden_size
        
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
        if not self.oned_input:
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
        else:
            x = self.input_block(x)
            x = self.res(x)
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
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None):
        super(ActorNetBase, self).__init__()

        self.disable_thinker = flags.wrapper_type == 1
        self.see_double = flags.return_double
        self.see_tree_rep = flags.see_tree_rep and not self.disable_thinker
        if self.see_tree_rep:
            self.tree_reps_shape = obs_space["tree_reps"].shape[1:]             
        self.see_h = flags.see_h and not self.disable_thinker
        if self.see_h:
            self.hs_shape = obs_space["hs"].shape[1:]
        self.see_x = flags.see_x
        if self.see_x and not self.disable_thinker:
            self.xs_shape = obs_space["xs"].shape[1:]
        self.see_real_state = flags.see_real_state        
        if flags.see_real_state:
            if obs_space["real_states"].dtype == 'uint8':
                self.state_dtype_n = 0
            elif obs_space["real_states"].dtype == 'float32':
                self.state_dtype_n = 1
            else:
                raise Exception(f"Unupported observation sapce", obs_space["real_states"])            
            low = torch.tensor(obs_space["real_states"].low)
            high = torch.tensor(obs_space["real_states"].high)
            self.need_norm = torch.isfinite(low).all() and torch.isfinite(high).all()            
            if self.need_norm:
                self.register_buffer("norm_low", low)
                self.register_buffer("norm_high", high)
            self.real_states_shape = obs_space["real_states"].shape[1:]

        if not self.disable_thinker:
            pri_action_space = action_space[0][0]            
        else:
            pri_action_space = action_space[0]

        if type(pri_action_space) == spaces.discrete.Discrete:    
            self.tuple_action = False                
            self.num_actions = pri_action_space.n    
            self.dim_actions = 1
            self.dim_rep_actions = self.num_actions
        else:
            self.tuple_action = True
            self.num_actions = pri_action_space[0].n    
            self.dim_actions = len(pri_action_space)    
            self.dim_rep_actions = self.dim_actions

        self.tran_dim = flags.tran_dim 
        self.tree_rep_rnn = flags.tree_rep_rnn and flags.see_tree_rep         
        self.se_lstm_table = flags.se_lstm_table and flags.see_tree_rep    
        self.x_rnn = flags.x_rnn and flags.see_x  
        self.real_state_rnn = flags.real_state_rnn and flags.see_real_state 

        self.num_rewards = 1
        self.num_rewards += int(flags.im_cost > 0.0)
        self.num_rewards += int(flags.cur_cost > 0.0)

        self.enc_type = flags.critic_enc_type  
        self.critic_zero_init = flags.critic_zero_init        

        self.sep_im_head = flags.sep_im_head
        self.last_layer_n = flags.last_layer_n

        self.float16 = flags.float16
        self.flags = flags        

        # encoder for state or encoding output
        last_out_size = self.dim_rep_actions + self.num_rewards
        if not self.disable_thinker:
            last_out_size += 2

        if self.see_h:
            self.h_encoder = AFrameEncoder(
                input_shape=self.hs_shape,                                 
                see_double=self.see_double
            )
            h_out_size = self.h_encoder.out_size
            last_out_size += h_out_size
        
        if self.see_x:
            self.x_encoder_pre = AFrameEncoder(
                input_shape=self.xs_shape, 
                downpool=True,
                firstpool=flags.x_enc_first_pool,
                see_double=self.see_double
            )
            x_out_size = self.x_encoder_pre.out_size
            if self.x_rnn:
                rnn_in_size = x_out_size
                self.x_encoder_rnn = RNNEncoder(
                    in_size=rnn_in_size,
                    flags=flags,
                )
                x_out_size = flags.tran_dim            
            last_out_size += x_out_size

        if self.see_real_state:
            self.real_state_encoder =  AFrameEncoder(
                input_shape=self.real_states_shape, 
                downpool=True,
                firstpool=flags.x_enc_first_pool,
                see_double=self.see_double
            )
            r_out_size = self.real_state_encoder.out_size
            if self.real_state_rnn:
                rnn_in_size = r_out_size
                self.r_encoder_rnn = RNNEncoder(
                    in_size=rnn_in_size,
                    flags=flags,
                )
                r_out_size = flags.tran_dim   
            last_out_size += r_out_size         
                    
        if self.see_tree_rep:            
            self.tree_rep_meaning = tree_rep_meaning
            in_size = self.tree_reps_shape[0]
            if self.se_lstm_table:
                assert flags.se_query_cur == 2                
                root_table_mask = torch.zeros(in_size, dtype=torch.bool)
                root_query_keys = [k for k in tree_rep_meaning if k.startswith("root_query_results")]
                for i in root_query_keys:
                    root_table_mask[self.tree_rep_meaning[i]] = 1        
                # print("root_query_size: ", sum(root_table_mask).long().item())        
                cur_table_mask = torch.zeros(in_size, dtype=torch.bool)
                cur_query_keys = [k for k in tree_rep_meaning if k.startswith("cur_query_results")]
                for i in cur_query_keys:
                    cur_table_mask[self.tree_rep_meaning[i]] = 1
                # print("cur_query_size: ", sum(cur_table_mask).long().item())        
                non_table_mask = torch.logical_or(root_table_mask, cur_table_mask)
                non_table_mask = torch.logical_not(non_table_mask)
                self.register_buffer("root_table_mask", root_table_mask)
                self.register_buffer("cur_table_mask", cur_table_mask)
                self.register_buffer("non_table_mask", non_table_mask)
                input_size = (sum(root_table_mask) / flags.se_query_size).long().item()
                self.tree_rep_table_lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=3, batch_first=True)
                in_size = torch.sum(non_table_mask).long() + 64 * 2

            if self.tree_rep_rnn:
                self.tree_rep_encoder = RNNEncoder(
                    in_size=in_size,
                    flags=flags
                )
                last_out_size += flags.tran_dim
            else:
                self.tree_rep_encoder = MLP(
                    input_size=in_size,
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

        self.policy = nn.Linear(last_out_size, self.num_actions * self.dim_actions)

        if not self.disable_thinker:
            if self.sep_im_head:
                self.im_policy = nn.Linear(last_out_size, self.num_actions * self.dim_actions)
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

        self.ordinal = flags.actor_ordinal
        if self.ordinal:
            indices = torch.arange(self.num_actions).view(-1, 1)
            ordinal_mask = (indices + indices.T) <= (self.num_actions - 1)
            ordinal_mask = ordinal_mask.float()
            self.register_buffer("ordinal_mask", ordinal_mask)

        self.initial_state(batch_size=1) # initialize self.state_idx


    def initial_state(self, batch_size, device=None):
        self.state_idx = {}
        idx = 0
        initial_state = ()
        
        conditions = [self.x_rnn, self.real_state_rnn, self.tree_rep_rnn]
        rnn_names = ["x_encoder_rnn", "r_encoder_rnn", "tree_rep_encoder"]
        state_names = ["x", "r", "tree_rep", "root_table"]

        for condition, rnn_name, state_name in zip(conditions, rnn_names, state_names):
            if condition:
                core_state = getattr(self, rnn_name).initial_state(batch_size, device=device)
                initial_state = initial_state + core_state
                self.state_idx[state_name] = slice(idx, idx+len(core_state))
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
        if not self.tuple_action: last_pri = last_pri.unsqueeze(-1)
        last_pri = util.encode_action(last_pri, self.dim_actions, self.num_actions)   
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
            tree_rep = env_out.tree_reps               
            tree_rep = torch.flatten(tree_rep, 0, 1)     

            if self.se_lstm_table:
                root_table = tree_rep[:, self.root_table_mask]
                root_table = torch.flip(root_table.view(T*B, self.flags.se_query_size, -1), dims=[1])
                root_table_rep, _ = self.tree_rep_table_lstm(root_table)
                root_table_rep = root_table_rep[:, -1]
                cur_table = tree_rep[:, self.cur_table_mask]
                cur_table = torch.flip(cur_table.view(T*B, self.flags.se_query_size, -1), dims=[1])
                cur_table_rep, _ = self.tree_rep_table_lstm(cur_table)
                cur_table_rep = cur_table_rep[:, -1]
                tree_rep = torch.concat([tree_rep[:, self.non_table_mask], root_table_rep, cur_table_rep], dim=-1)

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
                encoded_x, core_state_ = self.x_encoder_rnn(
                    encoded_x, done, core_state_)
                new_core_state[self.state_idx['x']] = core_state_
            
            final_out.append(encoded_x)

        if self.see_real_state:
            real_states = torch.flatten(env_out.real_states, 0, 1)   
            real_states = self.normalize(real_states)
            with autocast(enabled=self.float16):      
                encoded_real_state = self.real_state_encoder(real_states)
            if self.float16: encoded_real_state = encoded_real_state.float()

            if self.real_state_rnn:
                core_state_ = core_state[self.state_idx['r']]
                encoded_real_state, core_state_ = self.r_encoder_rnn(
                    encoded_real_state, done, core_state_)
                new_core_state[self.state_idx['r']] = core_state_

            final_out.append(encoded_real_state)

        final_out = torch.concat(final_out, dim=-1)   

        if self.last_layer_n > 0:
            final_out = self.final_mlp(final_out)     

        # compute logits
        pri_logits = self.policy(final_out)    
        pri_logits = pri_logits.view(T*B, self.dim_actions, self.num_actions)
        if self.ordinal: pri_logits = self.ordinal_encode(pri_logits)

        if not self.disable_thinker:
            if self.sep_im_head:
                im_logits = self.im_policy(final_out)                
                im_logits = im_logits.view(T*B, self.dim_actions, self.num_actions)
                if self.ordinal: im_logits = self.ordinal_encode(im_logits)
                im_mask = env_out.step_status <= 1 # imagainary action will be taken next
                im_mask = torch.flatten(im_mask, 0, 1).unsqueeze(-1).unsqueeze(-1)
                pri_logits = torch.where(im_mask, im_logits, pri_logits)
            reset_logits = self.reset(final_out)
        else:   
            reset_logits = None

        # compute entropy loss
        if compute_loss:
            entropy_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                input=pri_logits, target=F.softmax(pri_logits, dim=-1)
            )
            entropy_loss = torch.sum(entropy_loss, dim=-1)
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
        pri = self.sample(pri_logits, greedy=greedy, dim=-1)
        pri_logits = pri_logits.view(T, B, self.dim_actions, self.num_actions)
        pri = pri.view(T, B, self.dim_actions)        
        if not self.disable_thinker:
            reset = self.sample(reset_logits, greedy=greedy, dim=-1)
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
        pri_env = pri[-1, :, 0] if not self.tuple_action else pri[-1]        
        if not self.disable_thinker:
            action = (pri_env, reset[-1])            
        else:
            action = pri_env           
        action_prob = F.softmax(pri_logits, dim=-1)    
        if not self.tuple_action: action_prob = action_prob[:, :, 0]    

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
                1e-3 * torch.sum(pri_logits**2, dim=(-2,-1)) / 2
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
    
    def normalize(self, x):
        if self.state_dtype_n == 0: assert x.dtype == torch.uint8
        if self.state_dtype_n == 1: assert x.dtype == torch.float32
        if self.need_norm:
            x = (x.float() - self.norm_low) / \
                (self.norm_high -  self.norm_low)
        return x
    
    def ordinal_encode(self, logits):
        norm_softm = F.sigmoid(logits)
        norm_softm_tiled = torch.tile(norm_softm.unsqueeze(-1), [1,1,1,self.num_actions])
        return torch.sum(torch.log(norm_softm_tiled + 1e-8) * self.ordinal_mask + torch.log(1 - norm_softm_tiled + 1e-8) * (1 - self.ordinal_mask), dim=-1)

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
