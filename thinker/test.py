from collections import namedtuple
import sys

import time
import numpy as np
import argparse
import ray
import torch

from thinker.self_play import SelfPlayWorker
from thinker.learn_actor import ActorLearner
from thinker.buffer import *
import thinker.util as util
from thinker.net import *


flags = util.parse()
obs_shape=(3,80,80)
num_actions = 5
flags.model_rnn = True

cnet = ModelNet(obs_shape, num_actions, flags, rnn=True)
x = torch.rand(10, 16, 3, 80, 80)
actions = torch.zeros(10, 16).long()
done = torch.zeros(10, 16).bool()
state = cnet.core.init_state(16)

vs, logits, state = cnet.forward(x, actions, done, state, one_hot=False)
vs, logits, state = cnet.forward(x, actions, done, state, one_hot=False)
print(vs, logits)

sys.exit()

parser = define_parser()
flags = parser.parse_args([])

self_play_worker = SelfPlayWorker(None, None, 1, flags)
self_play_worker.gen_data()


sys.exit()


flags = util.parse()
flags.model_batch_size = 2
flags.model_unroll_length = 8
flags.model_k_step_return = 5
flags.actor_parallel_n = 4
flags.model_buffer_n = 1000
flags.model_warm_up_n = 500
flags.model_batch_mode = False

t = flags.model_unroll_length   
k = flags.model_k_step_return
n = flags.actor_parallel_n  

P = namedtuple("P", ["x","y"])
model_buffer = ModelBuffer(flags)

c = 0
for c in range(100):
    data = P(torch.full((t+k, n, 1),2*c), torch.full((t+k, n, 1),2*c+1))    
    model_buffer.write(data)
    r = model_buffer.read(1.)    
    if r is not None:
        data, weights, abs_flat_inds, ps_step = r
        #print(data.x[:,:,0]) 
        #model_buffer.update_priority(abs_flat_inds, np.zeros(flags.model_batch_size))

print("1 read", data.x[:,:,0])
model_buffer.update_priority(abs_flat_inds, np.full(flags.model_batch_size, 1000000))
data, weights, abs_flat_inds, ps_step = model_buffer.read(1.)   
print("2 read", data.x[:,:,0])

sys.exit()

parser = define_parser()
flags = parser.parse_args([])
env = Environment(flags)
actor_net = ActorNet(obs_shape=env.model_out_shape, num_actions=env.num_actions, flags=flags)
model_net = ModelNet(obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags)
model_net.train(False)

out = env.initial(model_net)
print(out)
a = torch.tensor([1,0,1]).long()
a = a.unsqueeze(0).unsqueeze(0)

core_state = actor_net.initial_state(1)
out = env.step(a, model_net)
print(out)
print(actor_net(out, core_state))