import torch
import gym
import numpy as np
from gym import wrappers
from display.config import mw_config
from display.policy import SAC
from copy import copy
import ipdb
import sys
import os
from display.envs.make_env import build_environment
from display.utils import param_to_policy


def display_model(ckpt, env_name):
    config = mw_config

    config.env_name = env_name + '-v2-goal-observable'
    envs = build_environment(config)

    config.device = ckpt.device

    ckpt_num = len(ckpt)
    print("{} parameters evaluate".format(ckpt_num))

    mean_reward_list = []
    mean_success_list = []
    mean_success_time_list = []

    seeds = config.seed

    for i, env in enumerate(envs):
        print("Now evaluate on {}, seed = {}".format(config.env_name, seeds[i]))
        agent = SAC(env.observation_space.shape[0], env.action_space, config)
        torch.manual_seed(seeds[i])
        avg_reward_list = []
        avg_success_list = []
        avg_success_time_list = []

        for i in range(ckpt_num):
            state_dict = param_to_policy(ckpt[i], agent.policy.state_dict())
            agent.policy.load_state_dict(state_dict)


            eval_reward_list = []
            eval_success_list = []
            eval_success_time_list = []

            for i in range(config.eval_episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                first_success_time = 0
                success = False
                rewards = []
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = env.step(action)
                    
                    if 'success' in info.keys():
                        success |= bool(info["success"])

                        if not success:
                            first_success_time += 1
                        
                    episode_reward += reward
                    state = next_state
                
                eval_success_list.append(success)
                eval_reward_list.append(episode_reward)
                eval_success_time_list.append(first_success_time)

            test_reward = np.average(eval_reward_list)
            test_success = np.average(eval_success_list)
            test_success_time = np.average(eval_success_time_list)
            print("----------------------------------------")
            print("Env: {}, Avg. Reward: {}, Avg. Success: {}, Avg Length: {}".format(config.env_name, round(test_reward, 2), round(test_success,2), round(test_success_time, 2)))
            print("----------------------------------------")

            avg_reward_list.append(test_reward)
            avg_success_list.append(test_success)
            avg_success_time_list.append(test_success_time)
        
        env.close()
        avg_reward_list.sort(reverse=True)
        avg_success_list.sort(reverse=True)
        avg_success_time_list.sort(reverse=False)

        mean_reward_list.append(avg_reward_list)
        mean_success_list.append(avg_success_list)
        mean_success_time_list.append(avg_success_time_list)

    reward_list = np.mean(mean_reward_list, axis=0)
    success_list = np.mean(mean_success_list, axis=0)
    success_time_list = np.mean(mean_success_time_list, axis=0)

    return reward_list, success_list, success_time_list
    

