import gymnasium 
import panda_gym
from gym.wrappers import TimeLimit
import ipdb
import numpy as np
    
class PandaWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
    def reset(self):
        state, info = self.env.reset()
        return state
    def step(self, action):
        next_state, reward, done, flag, info = self.env.step(action)
        return next_state, reward, done, info


class PandaNoiseWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
    def reset(self):
        state, info = self.env.reset()
        return state
    
    def step(self, action):
        size = self.env.action_space.shape[0]
        action = 0.1 * np.ones(size) + np.random.normal(0, np.sqrt(0.01), size=(size,)) + action
        next_state, reward, done, flag, info = self.env.step(action)
        return next_state, reward, done, info


def panda_env(env_name,control_type="joints", reward_type="sparse", render_mode="rgb_array", noise_type=False): 
    env = gymnasium.make(env_name, control_type=control_type, reward_type=reward_type, render_mode=render_mode)
    if noise_type:
        env = PandaNoiseWrapper(gymnasium.wrappers.FlattenObservation(env))
    else:
        env = PandaWrapper(gymnasium.wrappers.FlattenObservation(env))
    if 'Stack' in env_name:
        env = TimeLimit(env, max_episode_steps=50)
    else:
        env = TimeLimit(env, max_episode_steps=100)
    return env

if __name__ == "__main__":
    env = panda_make_env("PandaStack-v3")