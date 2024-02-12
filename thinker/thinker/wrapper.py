from collections import deque
import numpy as np
import cv2
import torch
import gym
from gym import spaces

class DummyWrapper(gym.Wrapper):
    """DummyWrapper that represents the core wrapper for the real env;
    the only function is to convert returning var into tensor
    and reset the env when it is done.
    """
    def __init__(self, env, env_n, flags, model_net, device=None, time=False):   
        gym.Wrapper.__init__(self, env)
        self.env_n = env_n
        self.flags = flags
        self.device = torch.device("cpu") if device is None else device 
        self.observation_space = spaces.Dict({
            "real_states": self.env.observation_space,
        })        
        if env.observation_space.dtype == 'uint8':
            self.state_dtype = torch.uint8
        elif env.observation_space.dtype == 'float32':
            self.state_dtype = torch.float32
        else:
            raise Exception(f"Unupported observation sapce", env.observation_space)

        low = torch.tensor(self.env.observation_space.low[0])
        high = torch.tensor(self.env.observation_space.high[0])
        self.need_norm = torch.isfinite(low).all() and torch.isfinite(high).all()
        self.norm_low = low.to(device)
        self.norm_high = high.to(device)

    def reset(self, model_net):
        obs = self.env.reset()
        obs_py = torch.tensor(obs, dtype=self.state_dtype, device=self.device)        
        states = {"real_states": obs_py}       
        return states 

    def step(self, action, model_net):  
        # action in shape (B, *) or (B,)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        obs, reward, done, info = self.env.step(action) 
        if np.any(done):
            done_idx = np.arange(self.env_n)[done]
            obs_reset = self.env.reset(done_idx)
            
        real_done = [m["real_done"] if "real_done" in m else done[n] for n, m in enumerate(info)]
        truncated_done = [m["truncated_done"] if "truncated_done" in m else False for n, m in enumerate(info)]
        cost = [m["cost"] if "cost" in m else False for n, m in enumerate(info)]
        obs_py = torch.tensor(obs, dtype=self.state_dtype, device=self.device)
        if np.any(done):
            obs_py[done] = torch.tensor(obs_reset, dtype=self.state_dtype, device=self.device)

        states = {
            "real_states": obs_py,
        }     
  
        info = {"real_done": torch.tensor(real_done, dtype=torch.bool, device=self.device),
                "truncated_done": torch.tensor(truncated_done, dtype=torch.bool, device=self.device),                
                "cost": torch.tensor(cost, dtype=torch.bool, device=self.device),                
                "step_status": torch.full((self.env_n,), fill_value=3, dtype=torch.long, device=self.device),
                }
        
        return (states, 
                torch.tensor(reward, dtype=torch.float32, device=self.device), 
                torch.tensor(done, dtype=torch.bool, device=self.device), 
                info)
    
    def unnormalize(self, x):
        assert x.dtype == torch.float or x.dtype == torch.float32
        if self.need_norm:
            ch = x.shape[-3]
            x = torch.clamp(x, 0, 1)
            x = x * (self.norm_high[-ch:] -  self.norm_low[-ch:]) + self.norm_low[-ch:]
        return x
    
    def normalize(self, x):
        if self.need_norm:            
            x = (x.float() - self.norm_low) / \
                (self.norm_high -  self.norm_low)
        return x
    
class PostWrapper(gym.Wrapper):
    """Wrapper for recording episode return, clipping rewards"""
    def __init__(self, env, reward_clip):
        gym.Wrapper.__init__(self, env)
        self.reset_called = False
        self.reward_clip = reward_clip
    
    def reset(self, model_net):
        state = self.env.reset(model_net)
        self.device = state["real_states"].device
        self.env_n = state["real_states"].shape[0]

        self.episode_step = torch.zeros(
            self.env_n, dtype=torch.long, device=self.device
        )

        self.episode_return = {}
        for key in ["re", "im", "cur"]:
            self.episode_return[key] = torch.zeros(
                self.env_n, dtype=torch.float, device=self.device
            )
        self.reset_called = True
        return state

    def step(self, action, model_net):
        assert self.reset_called, "need to call reset ONCE before step"
        state, reward, done, info = self.env.step(action, model_net)
        real_done = info["real_done"]

        self.episode_step += 1
        info["episode_step"] = self.episode_step.clone()
        self.episode_step[real_done] = 0

        self.episode_return["re"] += reward
        info["episode_return"] = self.episode_return["re"].clone()
        self.episode_return["re"][real_done] = 0.

        for prefix in ["im", "cur"]:
            if prefix+"_reward" in info:
                nan_mask = ~torch.isnan(info[prefix+"_reward"])
                self.episode_return[prefix][nan_mask] += info[prefix+"_reward"][nan_mask]
                info[prefix + "_episode_return"] = self.episode_return[prefix].clone()
                self.episode_return[prefix][real_done] = 0.
                if prefix == "im":
                    self.episode_return[prefix][info["step_status"] == 0] = 0.
        
        if self.reward_clip > 0.:
            reward = torch.clamp(reward, -self.reward_clip, +self.reward_clip)
        return state, reward, done, info
    
    def render(self, *args, **kwargs):  
        return self.env.render(*args, **kwargs)

def PreWrapper(env, name, grayscale=False, frame_wh=96, discrete_k=-1, repeat_action_n=0, one_to_threed_state=False):
    if discrete_k > 0: env = DiscretizeActionWrapper(env, K=discrete_k)
    if one_to_threed_state: env = TileObservationWrapper(env)

    if "Sokoban" in name:
        # sokoban
        env = TransposeWrap(env)
        if repeat_action_n > 0: env = RepeatActionWrapper(env, repeat_action_n=repeat_action_n)
    elif name.startswith("Safexp") or name.startswith("DM"):
        # safexp or DM control
        # assert discrete_k > 0, "Safeexp require discretizing the action space"
        if repeat_action_n > 0: env = RepeatActionWrapper(env, repeat_action_n=repeat_action_n)
    else:
        # atari
        env = StateWrapper(env)
        env = TimeLimit_(env, max_episode_steps=108000)
        env = NoopResetEnv(env, noop_max=30)
        if "NoFrameskip" in name:
            env = MaxAndSkipEnv(env, skip=4)
        env = wrap_deepmind(
            env,
            episode_life=True,
            clip_rewards=False,
            frame_stack=True,
            scale=False,
            grayscale=grayscale,
            frame_wh=frame_wh,
        )
        env = TransposeWrap(env)
    return env

# Standard wrappers

class TransposeWrap(gym.ObservationWrapper):
    """Image shape to channels x weight x height"""

    def __init__(self, env):
        super(TransposeWrap, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

class NoopWrapper(gym.Wrapper):
    def __init__(self, env, cost=0.0):
        gym.Wrapper.__init__(self, env)
        env.action_space.n += 1
        self.cost = cost

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # obs = obs[np.newaxis, :, :, :]
        self.last_obs = obs
        return obs

    def step(self, action):
        if action == 0:
            return self.last_obs, self.cost, False, {}
        else:
            obs, reward, done, info = self.env.step(action - 1)
            # obs = obs[np.newaxis, :, :, :]
            self.last_obs = obs
            return obs, reward, done, info

    def get_action_meanings(self):
        return [
            "NOOP",
        ] + self.env.get_action_meanings()

    def clone_state(self):
        state = self.env.clone_state()
        state["noop_last_obs"] = np.copy(self.last_obs)
        return state

    def restore_state(self, state):
        self.last_obs = np.copy(state["noop_last_obs"])
        self.env.restore_state(state)
        return

class TimeLimit_(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit_, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["truncated_done"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def clone_state(self):
        state = self.env.clone_state()
        state["timeLimit_elapsed_steps"] = self._elapsed_steps
        return state

    def restore_state(self, state):
        self._elapsed_steps = state["timeLimit_elapsed_steps"]
        self.env.restore_state(state)
        return

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=False, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

# Atari-related wrapped (taken from torchbeast)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = False
        self.was_done = False
        self.init = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        info["real_done"] = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        self.was_done = done
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done or not self.was_done:
            obs = self.env.reset(**kwargs)
            if not self.was_done and self.init:
                print("Warning: Resetting when episode is not done.")
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        self.init = True
        return obs

    def clone_state(self):
        state = self.env.clone_state()
        state["eps_life_vars"] = [self.lives, self.was_real_done]
        return state

    def restore_state(self, state):
        self.lives, self.was_real_done = state["eps_life_vars"]
        self.env.restore_state(state)
        return state

class DoneEnv(gym.Wrapper):
    def __init__(self, env):
        """Always done=True"""
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, True, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype,
        )
        self.frame_stack_n = k

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        # return np.concatenate(list(self.frames), axis=-1)
        return LazyFrames(list(self.frames))

    def clone_state(self):
        state = self.env.clone_state()
        state["frameStack"] = [np.copy(i) for i in self.frames]
        return state

    def restore_state(self, state):
        for i in state["frameStack"]:
            self.frames.append(i)
        self.env.restore_state(state)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

class StateWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def clone_state(self):
        state = self.env.clone_state()
        # state = self.env.clone_state(include_rng=True)
        return {"ale_state": state}

    def restore_state(self, state):
        # self.env.restore_state(state["ale_state"])
        self.env.restore_state(state["ale_state"])

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

def wrap_deepmind(
    env,
    episode_life=True,
    clip_rewards=True,
    frame_stack=False,
    scale=False,
    grayscale=False,
    frame_wh=96,
):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=frame_wh, height=frame_wh, grayscale=grayscale)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

class RepeatActionWrapper(gym.Wrapper):
    def __init__(self, env, repeat_action_n):
        super().__init__(env)
        self.repeat_action_n = repeat_action_n

        # Adjust observation space for stacked observations
        orig_obs_space = env.observation_space
        self.obs_shape = orig_obs_space.shape
        new_shape = (*self.obs_shape[:-1], self.obs_shape[-1] * repeat_action_n)
        self.observation_space = gym.spaces.Box(
            low=np.tile(orig_obs_space.low, repeat_action_n),
            high=np.tile(orig_obs_space.high, repeat_action_n),
            shape=new_shape,
            dtype=orig_obs_space.dtype
        )

        self.stacked_obs = np.zeros(new_shape, dtype=orig_obs_space.dtype)

    def reset(self):
        initial_obs = self.env.reset()
        self.stacked_obs = np.tile(initial_obs, self.repeat_action_n)
        return self.stacked_obs

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        for i in range(self.repeat_action_n):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            # Update the stacked observation
            start_index = i * self.obs_shape[-1]
            end_index = start_index + self.obs_shape[-1]
            self.stacked_obs[..., start_index:end_index] = obs

            if done:
                # Fill the remaining slots with the last observation if done
                for j in range(i + 1, self.repeat_action_n):
                    start_index = j * self.obs_shape[-1]
                    end_index = start_index + self.obs_shape[-1]
                    self.stacked_obs[..., start_index:end_index] = obs
                break

        return self.stacked_obs, total_reward, done, info

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, K=11):
        super().__init__(env)
        self.K = K  # Number of bins for discretization

        # Infer min and max actions from the original environment
        self.min_action = self.env.action_space.low
        self.max_action = self.env.action_space.high

        # Ensure the original action space is a Box
        assert isinstance(env.action_space, gym.spaces.Box), "The action space must be of type gym.spaces.Box"

        # Define the new action space as a tuple of Discrete(K) spaces
        self.action_space = spaces.Tuple([spaces.Discrete(K) for _ in range(env.action_space.shape[0])])

    def action(self, action):
        # Convert the discrete action to continuous action using vectorized operations
        action = np.array(action)  # Ensure action is a NumPy array
        discrete_to_cont = (action / (self.K - 1)) * (self.max_action - self.min_action) + self.min_action
        return discrete_to_cont
    
class TileObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TileObservationWrapper, self).__init__(env)

        # Ensure the original observation space is a Box with a single dimension
        assert isinstance(env.observation_space, gym.spaces.Box), "This wrapper only works with continuous state spaces (Box)"
        assert len(env.observation_space.shape) == 1, "The observation space must be 1D"

        n = env.observation_space.shape[0]

        # Update the observation space to Box(1, n, n)
        self.observation_space = spaces.Box(low=np.tile(env.observation_space.low, (1, n)).reshape(1, n, n),
                                     high=np.tile(env.observation_space.high, (1, n)).reshape(1, n, n),
                                     dtype=env.observation_space.dtype)

    def observation(self, observation):
        # Tile the observation and reshape to (1, n, n)
        n = len(observation)
        tiled_observation = np.tile(observation, (n, 1)).reshape(1, n, n)
        return tiled_observation
    
# wrapper for DM control
    
def convert_dm_control_to_gym_space(dm_control_space):    
    from dm_env import specs
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=dm_control_space.minimum, 
                           high=dm_control_space.maximum, 
                           dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'), 
                           high=float('inf'), 
                           shape=dm_control_space.shape, 
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                             for key, value in dm_control_space.items()})
        return space

class DMSuiteEnv(gym.Env):
    def __init__(self, domain_name, task_name, task_kwargs=None, environment_kwargs=None, visualize_reward=False, flatten=True, rgb=False):
        from dm_control import suite
        self.env = suite.load(domain_name, 
                              task_name, 
                              task_kwargs=task_kwargs, 
                              environment_kwargs=environment_kwargs, 
                              visualize_reward=visualize_reward)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0/self.env.control_timestep())}
        self.flatten = flatten
        self.rgb = rgb
        if self.rgb:
            # Assuming default resolution for simplicity; adjust as needed
            self.observation_space = spaces.Box(low=0, high=255, shape=(3, 80, 80), dtype=np.uint8)
        elif not flatten:
            self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        else:
            obs_spec = self.env.observation_spec()
            self.obs_dim = int(sum([np.prod(obs_spec[key].shape) for key in obs_spec]))  # Ensure obs_dim is an integer
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.viewer = None
    
    def seed(self, seed):
        return self.env.task.random.seed(seed)
    
    def step(self, action):
        timestep = self.env.step(action)
        if self.rgb:
            observation = self._get_rgb_image()
        elif not self.flatten:
            observation = timestep.observation
        else:        
            observation = self._flatten_observation(timestep.observation)
        reward = timestep.reward
        done = timestep.last()
        info = {'truncated_done': done}
        return observation, reward, done, info
    
    def reset(self):
        timestep = self.env.reset()
        if self.rgb:
            observation = self._get_rgb_image()
        elif not self.flatten:
            observation = timestep.observation
        else:        
            observation = self._flatten_observation(timestep.observation)
        return observation
    
    def render(self, mode='human', **kwargs):
        if 'camera_id' not in kwargs:
            kwargs['camera_id'] = 0  # Tracking camera
        use_opencv_renderer = kwargs.pop('use_opencv_renderer', False)
        
        img = self.env.physics.render(**kwargs)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                if not use_opencv_renderer:
                    from gym.envs.classic_control import rendering
                    self.viewer = rendering.SimpleImageViewer(maxwidth=1024)
                else:
                    from . import OpenCVImageViewer
                    self.viewer = OpenCVImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            raise NotImplementedError
        
    def _get_rgb_image(self):
        # Render the RGB image. Adjust width and height as necessary.
        img = self.env.physics.render(height=80, width=80, camera_id=0)  # Default camera; adjust if needed
        img = np.transpose(img, (2, 0, 1))  # Convert from HWC to CHW format expected in the observation space
        return img

        
    def _flatten_observation(self, observation):
        """Flatten an observation into a single vector."""
        return np.concatenate([observation[key].ravel() for key in observation])

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()