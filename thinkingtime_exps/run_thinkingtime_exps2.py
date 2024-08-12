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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run thinking time exps")
    parser.add_argument("--num_episodes", type=int, default=240)
    parser.add_argument("--env_name", type=str, default="cutoffpusht4")
    parser.add_argument("--num_thinking_steps", type=int, default=5)
    parser.add_argument("--gpu", type=bool, default=False)
    args = parser.parse_args()

    num_eps = args.num_episodes
    env_name = args.env_name
    num_thinking_steps = args.num_thinking_steps
    env_n = 1
    
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) # the default flags; almost all of them won't be used in DRC
    flags.mini = True
    flags.mini_unqtar = False
    flags.mini_unqbox = False
    gpu = True
    mini_sokoban = True
    results = []
    for num_thinking_steps in range(num_thinking_steps+1):
        print(f"==== ********** STEPS: {num_thinking_steps} ********** ====")
        for agent_name, d, n in [(f"{k}m", 3,3) for k in [10,20,30,40,50]]: #[(f"1{k}m", 3, 3) for k in [0,1,2,3,4,5,6,7,8,9]] + [(f"2{k}m", 3, 3) for k in [0,1,2,3,4,5]]:
            print(f"==== Running agent {agent_name} ====")
            ep_num = 0
            solve_count = 0
            all_count = 0

            env = thinker.make(
            f"Sokoban-{env_name}-v0", 
                env_n=env_n, 
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

            
            while all_count < num_eps:
                step_count = 0
                rnn_state = drc_net.initial_state(batch_size=env_n, device=env.device)
                env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)
                for _ in range(num_thinking_steps):
                    actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
                    state, reward, done, info = env.step(torch.tensor([0]))
                    env_out = util.create_env_out(torch.tensor([0]), state, reward, done, info, flags)
                actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
                state, reward, done, info = env.step(actor_out.action)
                step_count += 1

                while not done:
                    env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
                    with torch.no_grad():
                        actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)

                    state, reward, done, info = env.step(actor_out.action)
                    step_count += 1

                if step_count < 115-num_thinking_steps-1:
                    solve_count += 1
                all_count += 1
            results.append({"success_Rate": solve_count / num_eps, "agent": agent_name, "thinking_steps": num_thinking_steps})
            print(solve_count / num_eps)
    pd.DataFrame(results).to_csv(f"thinkingtime_results/{env_name}.csv")
