import gym
import robel
class RobelEnv(object):
    def __init__(self, env_name) -> None:
        self.env = gym.make(env_name)
    
    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps

class SparseRobelEnv(object):
    def __init__(self, env_name) -> None:
        self.env = gym.make(env_name)
    
    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, float(info["score/success"]), done, info
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps
    
if __name__ == "__main__":
	env_name = "DKittyWalkRandom-v0"
	env = SparseRobelEnv(env_name)
	env.reset()
	observation, reward, terminated, info = env.step(env.action_space.sample())
	import ipdb
	ipdb.set_trace()