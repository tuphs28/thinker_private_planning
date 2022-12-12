


sys.exit()

from thinker.train import define_parser
import ray
import time
import sys
import numpy as np
import argparse
import torch
import gym
from torch import nn

from thinker.train import ActorBuffer, ParamBuffer
from thinker.train import AB_CAN_WRITE, AB_FULL, AB_FINISH
from thinker.net import ActorNet, ModelNet
from thinker.env import Environment
from thinker.self_play import SelfPlayWorker

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