import gymnasium as gym
import numpy as np
from collections.abc import Mapping
import ipdb

class ContinuousTaskWrapper(gym.Wrapper):
	def __init__(self, env) -> None:
		super().__init__(env)

	def reset(self, *args, **kwargs):
		return super().reset(*args, **kwargs)

	def step(self, action):
		ob, rew, terminated, truncated, info = super().step(action)
		return ob, rew, False, truncated, info

class SuccessInfoWrapper(gym.Wrapper):
	def step(self, action):
		ob, rew, terminated, truncated, info = super().step(action)
		info["is_success"] = info["success"]
		return ob, rew, terminated, truncated, info

class AgentExtraFlattenWrapper(gym.ObservationWrapper):
	def __init__(self, env):
		super().__init__(env)
		# Calculate the flattened size for both 'agent' and 'extra' parts
		agent_space = self.flatten_space(env.observation_space['agent'])
		extra_space = self.flatten_space(env.observation_space['extra'])
		low = np.concatenate([agent_space.low, extra_space.low])
		high = np.concatenate([agent_space.high, extra_space.high])
		self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

	def flatten_space(self, space):
		"""Flatten a gym space into a single Box space."""
		if isinstance(space, gym.spaces.Box):
			return gym.spaces.Box(low=space.low.flatten(), high=space.high.flatten(), dtype=space.dtype)
		elif isinstance(space, gym.spaces.Dict) or isinstance(space, Mapping):
			low, high = [], []
			for key, subspace in space.spaces.items():
				if isinstance(subspace, gym.spaces.Box):
					low.extend(subspace.low.flatten())
					high.extend(subspace.high.flatten())
				else:
					raise NotImplementedError(f"Unsupported subspace type: {type(subspace)}")
			return gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
		else:
			raise NotImplementedError(f"Unsupported space: {type(space)}")

	def observation(self, observation):
		"""Extract and flatten 'agent' and 'extra' parts of the observation."""
		flat_agent_obs = self.flatten_individual_observation(observation['agent'])
		flat_extra_obs = self.flatten_individual_observation(observation['extra'])

		return np.concatenate([flat_agent_obs, flat_extra_obs])

	def flatten_individual_observation(self, obs_dict):
		"""Flatten an individual observation dictionary."""
		return np.concatenate([value.flatten() for value in obs_dict.values()])


# Define the environment creation function
def make_env(env_id):
	import mani_skill2.envs  # Import custom environments
	# import ipdb
	# ipdb.set_trace()
	env = gym.make(env_id, obs_mode="image", reward_mode="normalized_dense", control_mode="pd_ee_delta_pose", render_mode="rgb", max_episode_steps=200)
	env = ContinuousTaskWrapper(env)
	env = SuccessInfoWrapper(env)
	# env = FlattenObservationWrapper(env)
	env = AgentExtraFlattenWrapper(env)
	return env

if __name__ == "__main__":
	env_name = "Pour-v0"
	env = make_env(env_name)
	env.reset()
	num_steps = 0
	done = False
	# while not done:
	ipdb.set_trace()
	observation, reward, done, _, info = env.step(env.action_space.sample())
		# num_steps +=1
		# print("Steps:", num_steps)
	print(observation, reward, done, info)
