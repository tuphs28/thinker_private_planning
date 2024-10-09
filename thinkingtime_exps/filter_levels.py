import torch
import numpy as np
import thinker
import thinker.util as util
import gym
import gym_sokoban
import pandas as pd
import numpy as np
from thinker.actor_net import DRCNet
import os
import pandas as pd
import argparse

agent_name = "50m"
num_eps = 200
env_name = "med"
debug = True

flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) # the default flags; almost all of them won't be used in DRC
flags.mini = True
flags.mini_unqtar = False
flags.mini_unqbox = False
gpu = False
mini_sokoban = True
results = []

ep_num = 0
solve_count = 0
all_count = 0

filtered_ids = []
for j in range(1,200):

    base_solve, think_solve = 0, 0

    for num_thinking_steps in [0, 5]:
        env = thinker.make(
        f"Sokoban-{env_name}_clean_{j:04}-v0", 
            env_n=1, 
            gpu=gpu,
            wrapper_type=1, # wrapper_type 1 means default environment without Thinker-augmentation
            has_model=False, # the following arg are mainly for Thinker-augmentation only
            train_model=False, 
            parallel=False, 
            save_flags=False,
            mini=mini_sokoban, 
            mini_unqtar = False,
            mini_unqbox=False    
        ) 

        if j == 1 and num_thinking_steps == 0:
            flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) # the default flags; almost all of them won't be used in DRC
            flags.mini = True
            flags.mini_unqtar = False
            flags.mini_unqbox = False
            drc_net = DRCNet(
                obs_space=env.observation_space,
                action_space=env.action_space,
                flags=flags,
                record_state=True
                )
            drc_net.to(env.device)
            ckp = torch.load(f"../drc_mini/ckp_actor_realstep{agent_name}.tar", env.device)
            drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
        state = env.reset() 

        step_count = 0

        states = []
        rnn_state = drc_net.initial_state(batch_size=1, device=env.device)
        env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)
        for _ in range(num_thinking_steps):
            actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
            state, reward, done, info = env.step(torch.tensor([0]))
            env_out = util.create_env_out(torch.tensor([0]), state, reward, done, info, flags)
        actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
        state, reward, done, info = env.step(actor_out.action)
        step_count += 1

        while not done:
            states.append(state["real_states"][0])
            env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
            with torch.no_grad():
                actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)

            state, reward, done, info = env.step(actor_out.action)
            step_count += 1

        if step_count < 115-num_thinking_steps-1:
            if num_thinking_steps == 0:
                base_solve = 1
            elif num_thinking_steps == 5:
                think_solve = 1
        else:
            if num_thinking_steps == 0:
                base_solve = -1
            elif num_thinking_steps == 5:
                think_solve = -1

    print(f"{j=}, {base_solve=}, {think_solve=}")
    if think_solve == 1 and base_solve == -1:
        filtered_ids.append(j)

pd.DataFrame(filtered_ids).to_csv("filteredids3.csv")