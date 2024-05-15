from torch import flatten, cat
from thinker.actor_net import sample
from torch.nn.functional import relu, softmax, kl_div

class DRCTickLogitLens:
    def __init__(self, drc_net):
        self.drc_net = drc_net
    
    def activate(self):
        self.drc_net.record_core_output = True

    def deactivate(self):
        self.drc_net.record_core_output = False

    def get_logits(self, env_out):
        x = self.drc_net.normalize(env_out.real_states.float())
        x = flatten(x, 0, 1)
        x_enc = self.drc_net.encoder(x)
        core_output = self.drc_net.core.output
        N, B = core_output.shape[:2]
        core_output = flatten(self.drc_net.core.output, 0, 1)
        core_output = cat([cat([x_enc]*N,dim=0), core_output], dim=1)
        core_output = flatten(core_output, 1)
        final_out = relu(self.drc_net.final_layer(core_output))
        pri_logits = self.drc_net.policy(final_out)
        logits = pri_logits.view(N,B,5)
        return logits
    
    def sample_from_tick(self, env_out, tick, greedy=False):
        assert (tick > -1 and tick < self.drc_net.core.tran_t), f"Please enter a valid tick number betwen 0 and {self.drc_net.core.tran_t}"
        logits = self.get_logits(env_out)[tick,:,:]
        sampled_action = sample(logits, greedy)
        return sampled_action
    
    def get_avg_kl_div(self, env_out, ref_tick):
        logits = self.get_logits(env_out)
        log_probs = softmax(logits, dim=-1).log()
        return kl_div(input=log_probs[ref_tick,:,:], target=log_probs[-1,:,:], log_target=True, reduction="sum").item() / (logits.shape[1]*logits.shape[2])