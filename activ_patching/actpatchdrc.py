import torch
from thinker.actor_net import sample
from thinker.util import EnvOut
from typing import Optional

class ActPatchDRCNet:
    """
    Wrapper around DRCNet to patch in activations for different runs.
    """

    def __init__(self, drc_net, debug=False):
        self.drc_net = drc_net
        self.debug= debug

    def forward_normal(self, env_out, rnn_state):
        return self.drc_net(env_out, rnn_state)
    
    def forward_patch(self, env_out: EnvOut, rnn_state: tuple, greedy: bool = True, activ_type: Optional[str] = None, patch_dict: Optional[dict] = None, activ_ticks: Optional[list] = None, activs: Optional[torch.tensor] = None):
        """Run forward pass of wrapped DRCNet whilst patching in activations from activs

        Args:
            env_out (EnvOut): EnvOut object representing current state of environment
            rnn_state (tuple): current state of ConvLSTM - 2 tensors for each layer (e.g. h_t and c_t)
            greedy (bool, optional): whether to sample actions greedily. Defaults to True.
            activ_type (Optional[str], optional): an optional string that determines where to patching activations into - must be 'xenc', 'cell', 'hidden'- that is set to None if no patching is to occur. Defaults to None.
            patch_dict (Optional[dict], optional): a dictionary containing the channels to patch for the active_Type (hidden, cell, xenc) each layer e.g. if activ_type="hidden", to patch the first two channels of the hidden state for the first two layers {0: [0,1,] 1: [0,1]} . Defaults to None.
            activ_ticks (Optional[list], optional): list of ticks to patch over - if patching, must not be None. Defaults to None.
            activs (Optional[torch.tensor], optional): tensor of activations to patch in. Defaults to None.

        Returns:
            tuple: a tuple of (selected_action, action_probs, action_logits, updated_rnn_state)
        """
        
        activ_layers = list(patch_dict.keys())

        done = env_out.done
        T, B = done.shape
        x = self.drc_net.normalize(env_out.real_states.float())
        x = torch.flatten(x, 0, 1)
        x_enc = self.drc_net.encoder(x)
        core_input = x_enc.view(*((T, B) + x_enc.shape[1:]))

        assert len(core_input.shape) == 5
        core_output_list = []
        reset = done.float()
        if self.drc_net.record_state: 
            self.drc_net.core.hidden_state = []
            self.drc_net.core.hidden_state.append(torch.concat(rnn_state, dim=1)) 
        for n, (x_single, reset_single) in enumerate(
            zip(core_input.unbind(), reset.unbind())
        ):
            for t in range(self.drc_net.core.tran_t):

                if t > 0:
                    reset_single = torch.zeros_like(reset_single)
                reset_single = reset_single.view(-1)
                if activ_type is not None:
                    if t in activ_ticks and (0 in activ_layers or 1 in activ_layers or 2 in activ_layers):
                        if self.debug:
                            print(f"----- patching activations for tick {t} ---- ")
                        output, rnn_state = self.forward_single_patch(
                            x=x_single,
                            core_state=rnn_state,
                            reset=reset_single,
                            activ_type=activ_type, 
                            patch_dict=patch_dict,
                            activs=activs[:,t,:,:,:]
                        )
                    else:
                         output, rnn_state = self.drc_net.core.forward_single(
                            x_single, rnn_state, reset_single, reset_single
                        )
                else:
                    output, rnn_state = self.drc_net.core.forward_single(
                        x_single, rnn_state, reset_single, reset_single
                    )

                if self.drc_net.record_state: self.drc_net.core.hidden_state.append(torch.concat(rnn_state, dim=1))  

            core_output_list.append(output)

        core_output = torch.cat(core_output_list)
        if self.drc_net.record_state: 
           self.drc_net.core.hidden_state = torch.stack(self.drc_net.core.hidden_state, dim=1)

        core_output = torch.flatten(core_output, 0, 1)

        if activ_type == "xenc" and 3 in activ_layers and 2 in activ_ticks:
            if self.debug:
                print(f"--- Patching Layer 3 ---")
                print(f"patching channels {patch_channels} in xenc")
            patch_channels = patch_dict[3]
            x_enc[:,patch_channels,:,:] = activs[:,-1,[192+c for c in patch_channels],:,:]

        core_output = torch.cat([x_enc, core_output], dim=1)

        core_output = torch.flatten(core_output, 1)
        final_out = torch.nn.functional.relu(self.drc_net.final_layer(core_output))
        pri_logits = self.drc_net.policy(final_out)
        pri_logits = pri_logits.view(T*B, self.drc_net.dim_actions, self.drc_net.num_actions)
        pri_probs = torch.nn.functional.softmax(pri_logits.view(-1), dim=0)
        pri = sample(pri_logits, greedy=greedy, dim=-1)
        pri = pri.view(T, B, self.drc_net.dim_actions) 
        pri_env = pri[-1, :, 0] if not self.drc_net.tuple_action else pri[-1]   
        action = pri_env
        return (action, pri_probs, pri_logits.view(-1), rnn_state)
    
    def forward_single_patch(self, x, core_state, reset, activ_type=None, patch_dict={}, activs=None):
        reset = reset.float()

        activ_layers = list(patch_dict.keys())

        b, c, h, w = x.shape
        layer_n = 2
        out = core_state[(self.drc_net.core.num_layers - 1) * layer_n] * (1 - reset).view(
            b, 1, 1, 1
        )  # h_cur on last layer

        core_out = []
        new_core_state = []
        for n, cell in enumerate(self.drc_net.core.layers):
            cell_input = torch.concat([x, out], dim=1)
            h_cur = core_state[n * layer_n + 0] * (1 - reset.view(b, 1, 1, 1))
            c_cur = core_state[n * layer_n + 1] * (1 - reset.view(b, 1, 1, 1))
        
            if n in activ_layers and activ_type is not None:
                if self.debug:
                    print(f"--- Patching Layer {n} ---")
                patch_channels = patch_dict[n]
                if activ_type == "xenc":
                    patch_activs = activs[:,[192+c for c in patch_channels],:,:].detach().clone()
                elif activ_type == "hidden":
                    patch_activs = activs[:,[64*n+c for c in patch_channels],:,:].detach().clone()
                elif activ_type == "cell":
                    patch_activs = activs[:,[64*n+32+c for c in patch_channels],:,:].detach().clone()

                if activ_type == "xenc" and n in activ_layers:
                    if self.debug:
                        print(f"patching channels {patch_channels} in xenc")
                    cell_input[:,patch_channels,:,:] = patch_activs

                h_next, c_next = self.forward_cell_patch(
                    convlstm_cell=cell,
                    input=cell_input,
                    h_cur=h_cur,
                    c_cur=c_cur,
                    activ_type=activ_type,
                    patch_channels=patch_channels, 
                    patch_activs=patch_activs
                )
            else:
                if self.debug:
                    print(f"--- NOT patching layer {n} ---")
                h_next, c_next, _, _ = cell(
                    cell_input, h_cur, c_cur, None, None, None
                )
            if self.drc_net.core.grad_scale < 1 and h_next.requires_grad:
                h_next.register_hook(lambda grad: grad * self.drc_net.core.grad_scale)
                c_next.register_hook(lambda grad: grad * self.drc_net.core.grad_scale)
            new_core_state.append(h_next)
            new_core_state.append(c_next)
            out = h_next

        core_state = tuple(new_core_state)
        core_out = out.unsqueeze(0)
        return core_out, core_state
    
    def forward_cell_patch(self, convlstm_cell, input, h_cur, c_cur, activ_type=None, patch_channels=[], patch_activs=None):

        combined = torch.cat([input, h_cur], dim=1)  
        if convlstm_cell.pool_inject:
            combined = torch.cat(
                [combined, convlstm_cell.proj_max_mean(h_cur)], dim=1
            )  # concatenate along channel axis

        if convlstm_cell.linear:
            combined_conv = convlstm_cell.main(combined[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        else:
            combined_conv = convlstm_cell.main(combined)

        cc_i, cc_f, cc_o, cc_g, _ = torch.split(combined_conv, convlstm_cell.embed_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        if activ_type=="cell":
            if self.debug:
                print(f"patching channels {patch_channels} in cell")
            c_next[:,patch_channels,:,:] = patch_activs

        h_next = o * torch.tanh(c_next)
        if activ_type=="hidden":
            if self.debug:
                print(f"patching channels {patch_channels} in hidden")
            h_next[:,patch_channels,:,:] = patch_activs

        return h_next, c_next