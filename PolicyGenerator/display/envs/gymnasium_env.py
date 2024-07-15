import gym
import gymnasium 

class gymnasium2gymEnv(object):
	def __init__(self, env_name, seed, xml_file = None) -> None:
		if env_name.endswith("-v2"):
			self.env = gymnasium.make(env_name)
		else:  #* v3 and v4 take gymnasium.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale, etc.
			self.env = gymnasium.make(env_name, xml_file)
		self.seed = seed
	
	def reset(self):
		obs, info = self.env.reset(seed=self.seed)
		return obs

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		done =  terminated or truncated
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
	
if __name__ == "__main__":
	env_name = "Hopper-v4"
	env = gymnasium2gymEnv(env_name, seed=100)
	import ipdb
	ipdb.set_trace()
	env.reset()
	observation, reward, terminated, info = env.step(env.action_space.sample())
	print(observation, reward, terminated, info)
