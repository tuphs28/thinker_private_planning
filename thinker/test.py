from thinker.env import Environment
import thinker.util as util
import torch
import sys

flags = util.parse()
env = Environment(flags, model_wrap=False)
env_out = env.initial()
print("0:", env_out)
env_out = env.step(torch.tensor([3]).long())
print("1: env_out", env_out)
env_out = env.step(torch.tensor([2]).long())
print("2: env_out", env_out)



sys.exit()

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
flags.model_rnn = False

cnet = ModelNet(obs_shape, num_actions, flags)
x = torch.rand(16, 3, 80, 80)
actions = torch.zeros(1, 16).long()

rs, vs, logits, encodeds = cnet.forward(x, actions, one_hot=False)
print(vs.shape, logits.shape, encodeds)

sys.exit()

flags = util.parse()
flags.batch_size = 2
flags.model_unroll_length = 8
flags.model_k_step_return = 5
flags.actor_parallel_n = 4
flags.model_buffer_n = 1000

t = flags.model_unroll_length   
k = flags.model_k_step_return
n = flags.actor_parallel_n  

P = namedtuple("P", ["x","y"])
model_buffer = ModelBuffer(flags)

c = 0
for c in range(100):
    data = P(np.full((t+k, n, 1),2*c), np.full((t+k, n, 1),2*c+1))    
    model_buffer.write(data)
    data, weights, abs_flat_inds = model_buffer.read(1.)    
    print(data) 
    model_buffer.update_priority(abs_flat_inds, np.zeros(flags.batch_size))

model_buffer.update_priority(abs_flat_inds, np.full(flags.batch_size, 100000))   

sys.exit()
parser = define_parser()
flags = parser.parse_args([])

self_play_worker = SelfPlayWorker(None, None, 1, flags)
self_play_worker.gen_data()


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