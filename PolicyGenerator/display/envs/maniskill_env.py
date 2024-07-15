import gym
import gymnasium 
import numpy as np
import mani_skill2.envs

# env = gymnasium.make("Pour-v0",obs_mode='state',control_mode='pd_ee_delta_pos')
class ManiSkillEnv(object):
	def __init__(self, env_name, seed, obs_mode='state',control_mode='pd_ee_delta_pos') -> None:
		self.env = gymnasium.make(env_name, obs_mode=obs_mode, control_mode = control_mode)
		self.seed = seed
	
	def reset(self):
		obs, info = self.env.reset(seed=self.seed)
		return obs

	def step(self, action):
		obs, reward, done, flag, info = self.env.step(action)
		return obs, reward, flag, info
	
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
	env_name = "PickCube-v0"
	env = ManiSkillEnv(env_name, seed=100)
	env.reset()
	num_steps = 0
	done = False
	while not done:
		observation, reward, done, info = env.step(env.action_space.sample())
		num_steps +=1
		print("Steps:", num_steps)
	print(observation, reward, done, info)
