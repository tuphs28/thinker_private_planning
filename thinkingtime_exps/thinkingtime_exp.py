import thinker
from thinker.actor_net import DRCNet
import thinker.util as util
import torch
import pandas

results = []
num_eps = 240
env_n = 1
gpu = False
mini_sokoban = True
for num_thinking_steps in range(8):
    step_results = {}
    print(f"==== ********** STEPS: {num_thinking_steps} ********** ====")
    for agent_name, d, n in [(f"{k}m", 3,3) for k in [16,17,18,19,20]]: #[(f"1{k}m", 3, 3) for k in [0,1,2,3,4,5,6,7,8,9]] + [(f"2{k}m", 3, 3) for k in [0,1,2,3,4,5]]:
        print(f"==== Running agent {agent_name} ====")
        ep_num = 0
        levs = []
        env = thinker.make(
        "Sokoban-cutoffpusht4-v0", 
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
            record_state=True,
            d=d,
            t=n
            )
        drc_net.to(env.device)
        ckp = torch.load(f"../drc_mini/ckp_actor_realstep{agent_name}.tar", env.device)
        drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)

        solve_count = 0
        all_count = 0
        step_count = 0
        while all_count < num_eps:
            rnn_state = drc_net.initial_state(batch_size=env_n, device=env.device)
            state = env.reset() 
            for _ in range(num_thinking_steps-1):
                env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False) # this converts the state to EnvOut object that can be processed by actor
                actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
                state, reward, done, info = env.step(torch.tensor([0]))
            env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
            actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
            state, reward, done, info = env.step(torch.tensor([0]))
            while not done:
                env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
                with torch.no_grad():
                    actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)

                agent_loc = (state["real_states"][0][4] == 1).to(int).argmax() 
                thinking_mode = False
                agent_x, agent_y = agent_loc % 8, (agent_loc -(agent_loc % 8))//8

                state, reward, done, info = env.step(actor_out.action)
                step_count += 1

                if done:
                    if step_count < 110:
                        solve_count += 1
                        levs.append(all_count)
                    all_count += 1
                    step_count = 0
        step_results[agent_name] = solve_count / num_eps
        print(solve_count / num_eps)
    results.append(step_results)