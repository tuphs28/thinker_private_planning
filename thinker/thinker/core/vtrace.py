# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections

import torch
import torch.nn.functional as F


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
        "norm_stat"
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages norm_stat")


def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions)

@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,    
    rewards,
    values,
    values_enc,
    rv_tran,
    enc_type,
    bootstrap_value,
    flags,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    lamb=1.0,
    norm_stat=None
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = lamb * torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        target_values = rewards + discounts * vs_t_plus_1
        if flags.return_norm_type == -1:
            norm_stat = None
        else:
            if flags.return_norm_type == 0:
                norm_v = target_values 
            elif flags.return_norm_type == 1:
                norm_v = clipped_pg_rhos * (target_values - values)

            if flags.return_norm_buffer_n <= 0:
                if not flags.return_norm_use_std:
                    new_lq = torch.quantile(norm_v, 1 - flags.return_norm_per).detach()
                    new_uq = torch.quantile(norm_v, flags.return_norm_per).detach()            
                else:
                    new_lq = 0
                    new_uq = torch.sqrt(torch.mean(torch.square(norm_v)))
                if norm_stat is None:
                    norm_stat = (new_lq, new_uq)
                else:
                    norm_stat = (norm_stat[0]*flags.return_norm_decay + new_lq*(1-flags.return_norm_decay),
                                    norm_stat[1]*flags.return_norm_decay + new_uq*(1-flags.return_norm_decay),)
                norm_factor = torch.clamp(norm_stat[1] - norm_stat[0], min=flags.return_norm_b)               
                norm_stat = norm_stat + (norm_factor,)
            else:
                if norm_stat is None:
                    buffer = FifoBuffer(flags.return_norm_buffer_n, device=target_values.device)
                else:
                    buffer = norm_stat[-1]
                buffer.push(norm_v)
                if not flags.return_norm_use_std:
                    lq = buffer.get_percentile(1 - flags.return_norm_per)
                    uq = buffer.get_percentile(flags.return_norm_per)
                else:
                    lq = 0.
                    uq = torch.sqrt(buffer.get_variance())                
                norm_stat = (lq, uq,)          
                norm_factor = torch.clamp(norm_stat[1] - norm_stat[0], min=flags.return_norm_b)               
                norm_stat = norm_stat + (norm_factor, buffer)
        
        if enc_type in [0, 4] or not flags.return_norm_type == -1:
            pg_advantages = clipped_pg_rhos * (target_values - values)
            if not flags.return_norm_type == -1:
                #pg_advantages = torch.clamp(pg_advantages, norm_stat[0], norm_stat[1])
                pg_advantages = pg_advantages / norm_factor                
        elif enc_type == 1:
            pg_advantages = clipped_pg_rhos * (rv_tran.encode(target_values) - values_enc)
        elif enc_type in [2, 3]:
            pg_advantages = clipped_pg_rhos * (rv_tran.encode(target_values) - 
                                               rv_tran.encode_s(rv_tran.decode(values_enc)))
        else:
            raise Exception("Unknown reward type: ", rv_tran)
        

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages, norm_stat=norm_stat)

class FifoBuffer:
    def __init__(self, size, device):
        self.size = size
        self.buffer = torch.empty((self.size,), dtype=torch.float32, device=device).fill_(float('nan'))
        self.current_index = 0
        self.num_elements = 0
    
    def push(self, data):
        t, b = data.shape
        num_entries = t * b
        assert num_entries <= self.size, "Data too large for buffer"
        
        start_index = self.current_index
        end_index = (self.current_index + num_entries) % self.size
        
        if end_index < start_index:
            # The new data wraps around the buffer
            remaining_space = self.size - start_index
            self.buffer[start_index:] = data.flatten()[:remaining_space]
            self.buffer[:end_index] = data.flatten()[remaining_space:]
        else:
            # The new data fits within the remaining space
            self.buffer[start_index:end_index] = data.flatten()
        
        self.current_index = end_index
        self.num_elements = min(self.num_elements + num_entries, self.size)
    
    def get_percentile(self, percentile):
        num_valid_elements = min(self.num_elements, self.size)
        if num_valid_elements == 0: return None
        return torch.quantile(self.buffer[:num_valid_elements], q=percentile)
    
    def get_variance(self):
        num_valid_elements = min(self.num_elements, self.size)
        if num_valid_elements == 0: return None
        return torch.mean(torch.square(self.buffer[:num_valid_elements]))
    
    def get_mean(self):
        num_valid_elements = min(self.num_elements, self.size)
        if num_valid_elements == 0: return None
        return torch.mean(self.buffer[:num_valid_elements])
    
    def full(self):
        return self.num_elements >= self.size

