# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment class for MonoBeast."""

import torch

def _format_frame(frame, bsz=None):
    frame = torch.from_numpy(frame)
    if bsz is not None:
        return frame.view((1,) + frame.shape)
    else:
        return frame.view((1, 1) + frame.shape)

class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        initial_frame = _format_frame(self.gym_env.reset())
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        frame, reward, done, unused_info = self.gym_env.step(action.item())     
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        truncated_done = 'TimeLimit.truncated' in unused_info and unused_info['TimeLimit.truncated']
        truncated_done = torch.tensor(truncated_done).view(1, 1)
        return dict(
            frame=frame,
            reward=reward,
            done=done,
            truncated_done=truncated_done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )

    def close(self):
        self.gym_env.close()

    def clone_state(self):
        state = [self.episode_return.clone(), self.episode_step.clone()]
        state.append(self.gym_env.clone_state())
        return state
        
    def restore_state(self, state):
        self.episode_return = state[0].clone()
        self.episode_step = state[1].clone()
        self.gym_env.restore_state(state[2])

class Vec_Environment:
    def __init__(self, gym_env, bsz):
        self.gym_env = gym_env
        self.bsz = bsz
        self.episode_return = torch.zeros(1, self.bsz)
        self.episode_step = torch.zeros(1, self.bsz)        

    def initial(self):
        initial_reward = torch.zeros(1, self.bsz)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, self.bsz, dtype=torch.int64)
        self.episode_return = torch.zeros(1, self.bsz)
        self.episode_step = torch.zeros(1, self.bsz, dtype=torch.int32)
        initial_done = torch.ones(1, self.bsz, dtype=torch.uint8)
        initial_frame = _format_frame(self.gym_env.reset(), self.bsz)
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        frame, reward, done, unused_info = self.gym_env.step(action.detach().cpu().numpy())   
        
        self.episode_step = self.episode_step + 1
        self.episode_return = self.episode_return + torch.Tensor(reward).unsqueeze(0)
        episode_step = self.episode_step
        episode_return = self.episode_return
        
        done = torch.tensor(done).view(1, self.bsz)
        truncated_done = ['TimeLimit.truncated' in x and x['TimeLimit.truncated'] for x in unused_info]
        truncated_done = torch.tensor(truncated_done).view(1, self.bsz)
        
        self.episode_return = (~done).float() * self.episode_return
        self.episode_step = (~done).float() * self.episode_step
        
        frame = _format_frame(frame, self.bsz)
        reward = torch.tensor(reward).view(1, self.bsz).float()
        
        return dict(
            frame=frame,
            reward=reward,
            done=done,
            truncated_done=truncated_done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action.unsqueeze(0),
        )
    
    def clone_state(self):
        state = [self.episode_return.clone(), self.episode_step.clone()]
        for k in self.gym_env.envs: 
            state.append(k.clone_state())
        return state
        
    def restore_state(self, state):
        self.episode_return = state[0].clone()
        self.episode_step = state[1].clone()
        for n, k in enumerate(self.gym_env.envs): 
            k.restore_state(state[2+n])

    def close(self):
        self.gym_env.close()