from collections import deque
import numpy as np
import cv2
import torch
import gym
from gym import spaces
import thinker.util as util
import time

class DummyWrapper(gym.Wrapper):
    """DummyWrapper that represents the core wrapper for the real env;
    the only function is to convert returning var into tensor
    and reset the env when it is done.
    """
    def __init__(self, env, env_n, flags, model_net, device=None, timing=False):   
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

        self.train_model = self.flags.train_model
        action_space =  env.action_space[0]
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(action_space)

    def reset(self, model_net):
        obs = self.env.reset()
        obs_py = torch.tensor(obs, dtype=self.state_dtype, device=self.device)                
        if self.train_model: 
            self.per_state = model_net.initial_state(batch_size=self.env_n, device=self.device)
            pri_action = torch.zeros(self.env_n, self.dim_actions, dtype=torch.long, device=self.device)
            done = torch.zeros(self.env_n, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                model_net_out = model_net(
                    x=obs_py, 
                    done=done,
                    actions=pri_action.unsqueeze(0), 
                    state=self.per_state,
                    one_hot=False)       
            self.per_state = model_net_out.state
            self.baseline = model_net_out.vs[-1]
        states = {"real_states": obs_py}       
        return states 

    def step(self, action, model_net):  
        # action in shape (B, *) or (B,)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()        

        obs, reward, done, info = self.env.step(action) 
        if np.any(done):
            done_idx = np.arange(self.env_n)[done]
            obs_reset = self.env.reset(idx=done_idx)
            
        obs_py = torch.tensor(obs, dtype=self.state_dtype, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)
        if torch.any(done):
            obs_py[done] = torch.tensor(obs_reset, dtype=self.state_dtype, device=self.device)
        states = {
            "real_states": obs_py,
        }     

        info = util.dict_map(info, lambda x: torch.tensor(x, device=self.device))
        info["step_status"] = torch.full((self.env_n,), fill_value=3, dtype=torch.long, device=self.device)
        
        if self.train_model:             
            info["initial_per_state"] = self.per_state
            info["baseline"] = self.baseline
            pri_action = torch.tensor(action, dtype=torch.long, device=self.device)
            if not self.tuple_action: pri_action = pri_action.unsqueeze(-1)          
            with torch.no_grad():
                model_net_out = model_net(
                    x=obs_py, 
                    done=done,
                    actions=pri_action.unsqueeze(0), 
                    state=self.per_state,
                    one_hot=False)       
                self.per_state = model_net_out.state
                self.baseline = model_net_out.vs[-1]
        
        return states, reward, done, info
    
class PostWrapper(gym.Wrapper):
    """Wrapper for recording episode return, clipping rewards"""
    def __init__(self, env, flags, device):
        gym.Wrapper.__init__(self, env)
        self.reset_called = False        
        low = torch.tensor(self.env.observation_space["real_states"].low[0])
        high = torch.tensor(self.env.observation_space["real_states"].high[0])
        self.need_norm = torch.isfinite(low).all() and torch.isfinite(high).all()
        self.norm_low = low
        self.norm_high = high

        self.disable_thinker = flags.wrapper_type == 1
        if not self.disable_thinker:
            self.pri_action_space = self.env.action_space[0][0]            
        else:
            self.pri_action_space = self.env.action_space[0]
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(self.pri_action_space)
        if not self.discrete_action:
            self.action_space_low = torch.tensor(self.pri_action_space.low, dypte=torch.float, device=device)
            self.action_space_high = torch.tensor(self.pri_action_space.high, dypte=torch.float, device=device)
    
    def reset(self, model_net):
        state = self.env.reset(model_net)
        self.device = state["real_states"].device
        self.env_n = state["real_states"].shape[0]

        self.episode_step = torch.zeros(
            self.env_n, dtype=torch.long, device=self.device
        )

        self.episode_return = {}
        for key in ["im", "cur"]:
            self.episode_return[key] = torch.zeros(
                self.env_n, dtype=torch.float, device=self.device
            )
        self.reset_called = True
        return state

    def step(self, action, model_net):
        assert self.reset_called, "need to call reset ONCE before step"

        if not self.discrete_action:            
            if not self.disable_thinker:
                pri_action, reset_action = action
                pri_action = torch.clamp(pri_action, self.action_space_low, self.action_space_high)
                action = (pri_action, reset_action)
            else:
                action = torch.clamp(action, self.action_space_low, self.action_space_high)

        state, reward, done, info = self.env.step(action, model_net)
        real_done = info["real_done"]        

        for prefix in ["im", "cur"]:
            if prefix+"_reward" in info:
                nan_mask = ~torch.isnan(info[prefix+"_reward"])
                self.episode_return[prefix][nan_mask] += info[prefix+"_reward"][nan_mask]
                info[prefix + "_episode_return"] = self.episode_return[prefix].clone()
                self.episode_return[prefix][real_done] = 0.
                if prefix == "im":
                    self.episode_return[prefix][info["step_status"] == 0] = 0.        
        return state, reward, done, info
    
    def render(self, *args, **kwargs):  
        return self.env.render(*args, **kwargs)    
    
    def unnormalize(self, x):
        assert x.dtype == torch.float or x.dtype == torch.float32
        if self.need_norm:
            ch = x.shape[-3]
            x = torch.clamp(x, 0, 1)
            x = x * (self.norm_high[-ch:] -  self.norm_low[-ch:]) + self.norm_low[-ch:]
        return x
    
    def normalize(self, x):
        if self.need_norm:    
            if self.norm_low.device != x.device or self.norm_high.device != x.device:
                self.norm_low = self.norm_low.to(x.device)
                self.norm_high = self.norm_high.to(x.device)
            x = (x.float() - self.norm_low) / (self.norm_high -  self.norm_low)
        return x

def PreWrapper(env, name, grayscale=False, frame_wh=96, discrete_k=-1, repeat_action_n=0, atari=False):
    if discrete_k > 0: env = DiscretizeActionWrapper(env, K=discrete_k)
    if repeat_action_n > 0: env = RepeatActionWrapper(env, repeat_action_n=repeat_action_n)        
    if "NoFrameskip" in name and not atari: 
        raise Exception(f"{name} is likely an Atari game but flags.Atari is False")
    if atari: 
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
    if env.observation_space.dtype == np.float64:
        env = ScaledFloatFrame(env)
    if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 3:
        # 3d input, need transpose
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
        """
        An environment wrapper that scales observations from uint8 to float32 and
        normalizes them if they are uint8. If observations are float64, it converts them to float32 without normalization.
        """
        super(ScaledFloatFrame, self).__init__(env)
        assert self.env.observation_space.dtype in [np.uint8, np.float64]

        # Determine if the original observation space is uint8
        self.is_uint8 = self.env.observation_space.dtype == np.uint8
        # Adjust the observation space to reflect the change in dtype
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low if not self.is_uint8 else 0,
            high=self.env.observation_space.high if not self.is_uint8 else 1,
            shape=self.env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        # Convert observation to float32
        observation = np.array(observation, dtype=np.float32)        
        # Normalize only if the original observation space was uint8
        if self.is_uint8: observation = observation / 255.0
        return observation

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


# wrapper for normalization

class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

# wrapper after parallel env but before core wrapper

# https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py
def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action, **kwargs):
        obs, rews, dones, infos = self.env.step(action, **kwargs)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, dones, infos    

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs)
        else:
            return self.normalize(np.array([obs]))[0]

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)   
    
    def clone_state(self, idx=None):
        state = self.env.clone_state(idx)
        if idx is None: idx = range(self.num_envs)
        for i in idx:
            state[i]["obs_mean"] = self.obs_rms.mean
            state[i]["obs_var"] = self.obs_rms.var
            state[i]["obs_count"] = self.obs_rms.count
        return state
    
    def restore_state(self, state, idx=None):
        self.env.restore_state(state, idx)
        if idx is None: idx = range(self.num_envs)        
        self.obs_rms.mean = state[0]["obs_mean"]
        self.obs_rms.var = state[0]["obs_var"]
        self.obs_rms.count = state[0]["obs_count"]

class NormalizeReward(gym.core.Wrapper):
    def __init__(
        self,
        env,
        gamma=0.99,
        epsilon=1e-8,
    ):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action, **kwargs):
        obs, rews, dones, infos = self.env.step(action, **kwargs)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, dones, infos

    def normalize(self, rews):
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
    
    def clone_state(self, idx=None):
        state = self.env.clone_state(idx)
        if idx is None: idx = range(self.num_envs)
        for i in idx:
            state[i]["return_mean"] = self.return_rms.mean
            state[i]["return_var"] = self.return_rms.var
            state[i]["return_count"] = self.return_rms.count
        return state
    
    def restore_state(self, state, idx=None):
        self.env.restore_state(state, idx)
        if idx is None: idx = range(self.num_envs)        
        self.return_rms.mean = state[0]["return_mean"]
        self.return_rms.var = state[0]["return_var"]
        self.return_rms.count = state[0]["return_count"]
    
class InfoConcat(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.is_vector_env
        self.num_envs = getattr(env, "num_envs", 1)        

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs) 
        real_done = np.array([m["real_done"] if "real_done" in m else done[n] for n, m in enumerate(info)], dtype=np.bool_)
        truncated_done = np.array([m["truncated_done"] if "truncated_done" in m else False for n, m in enumerate(info)], dtype=np.bool_)
        cost = np.array([m["cost"] if "cost" in m else False for n, m in enumerate(info)], dtype=np.bool_)
        info = {
            "real_done": real_done,
            "truncated_done": truncated_done,
            "cost": cost,
        }
        return obs, reward, done, info
    
    def default_info(self):
        info = {
            "real_done": np.zeros(self.num_envs, dtype=np.bool_),
            "truncated_done": np.zeros(self.num_envs, dtype=np.bool_),
            "cost": np.zeros(self.num_envs, dtype=np.bool_),
        }
        return info
    
    def clone_state(self, idx=None):
        return self.env.clone_state(idx)
    
    def restore_state(self, state, idx=None):
        return self.env.restore_state(state, idx)

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)        
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_step = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, **kwargs):
        idx = kwargs.get("idx", None)     
        if idx is None:
            self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self.episode_step = np.zeros(self.num_envs, dtype=np.int64)
        else:
            self.episode_returns[idx] = 0.
            self.episode_step[idx] = 0

        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):        
        idx = kwargs.get("idx", None)     
        obs, reward, done, info = self.env.step(action, **kwargs)
        real_done = info["real_done"]
        if idx is None:
            self.episode_return = self.episode_return + reward
            self.episode_step = self.episode_step + 1
        else:
            self.episode_return[idx] = self.episode_return[idx] + reward
            self.episode_step[idx] = self.episode_step[idx] + 1
        episode_return = self.episode_return
        episode_step = self.episode_step

        if np.any(real_done):
            episode_return = np.copy(episode_return)
            episode_step = np.copy(episode_step)

            if idx is None:    
                self.episode_return[real_done] = 0.
                self.episode_step[real_done] = 0
            else:
                idx_b = np.zeros(self.num_envs, np.bool_)
                idx_b[idx] = True
                self.episode_return[idx_b & real_done] = 0.
                self.episode_step[idx_b & real_done] = 0
                
        info["episode_return"] = episode_return
        info["episode_step"] = episode_step
        return obs, reward, done, info
    
    def default_info(self):
        info = self.env.default_info()
        info["episode_return"] =  np.zeros(self.num_envs, dtype=np.float32)
        info["episode_step"] =  np.zeros(self.num_envs, dtype=np.int64)
        return info
    
    def clone_state(self, idx=None):
        state = self.env.clone_state(idx)
        if idx is None: idx = range(self.num_envs)
        for n, i in enumerate(idx):
            state[n]["episode_return"] = self.episode_return[i]
            state[n]["episode_step"] = self.episode_step[i]
        return state
    
    def restore_state(self, state, idx=None):
        self.env.restore_state(state, idx)
        if idx is None: idx = range(self.num_envs)
        for n, i in enumerate(idx):
            self.episode_return[i] = state[n]["episode_return"]
            self.episode_step[i] = state[n]["episode_step"]

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
    
    def reset(self, seed=None, options=None):
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