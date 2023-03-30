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


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    flags,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    lamb=1.0,
    norm_stat=None,    
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        flags=flags,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        lamb=lamb,
        norm_stat=norm_stat
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,    
    rewards,
    values,
    values_enc_s,
    reward_tran,
    bootstrap_value,
    flags,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    lamb=1.0,
    return_norm=False,
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
        if flags.return_norm:
            if flags.return_norm_type == 0:
                norm_v = target_values 
            elif flags.return_norm_type == 1:
                norm_v = clipped_pg_rhos * (target_values - values)
            new_lq = torch.quantile(norm_v,  0.05).detach()
            new_uq = torch.quantile(norm_v,  0.95).detach()            
            if norm_stat is None:
                norm_stat = (new_lq, new_uq)
            else:
                norm_stat = (norm_stat[0]*0.99 + new_lq*0.01,
                             norm_stat[1]*0.99 + new_uq*0.01,)
            norm_factor = torch.max(norm_stat[1] - norm_stat[0], torch.tensor([
                flags.return_norm_b], device=target_values.device))        
        else:
            norm_stat = None

        if reward_tran is None or flags.return_norm:
            pg_advantages = clipped_pg_rhos * (target_values - values)
            if return_norm:
                pg_advantages = pg_advantages / norm_factor
        else:
            pg_advantages = clipped_pg_rhos * (reward_tran.encode(target_values) - values_enc_s)
        

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages, norm_stat=norm_stat)
