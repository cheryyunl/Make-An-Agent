if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
import torch
import numpy as np
import copy
import random
import wandb
import tqdm
import time
import dill
from omegaconf import OmegaConf
import datetime
from hydra.core.hydra_config import HydraConfig
from dataset import Dataset
from policy_generator.model.common.normalizer import LinearNormalizer
from policy_generator.model.encoder.param_encoder import EncoderDecoder
from policy_generator.policy.policy_generator import PolicyGenerator
from display.display_policy import display_model


class EvalWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg

        self._saving_thread = None

        # set seed
        seed = cfg.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: PolicyGenerator  = hydra.utils.instantiate(cfg.policy)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.normalizer = LinearNormalizer()

        ckpt_path = cfg.eval.ckpt_dir
        encoder_path =  cfg.eval.encoder_dir

        self.param_encoder = EncoderDecoder(1024, 1, 1, 0.0001, 0.001)
        self.load_checkpoint(ckpt_path, evaluate=True)
        self.load_encoder(encoder_path, evaluate=True)

        self.model.set_normalizer(self.normalizer)
        self.model.eval()

    def rollout(self, data, env):
        nparam = data['param']
        ntraj = data['traj']
        ntask = data['task']


        eval_dict = {'traj': ntraj, 'task': ntask}
        pred_param = self.model.predict_paremeters(eval_dict)
        nparam = nparam.reshape(-1, 2, 1024)
        param = self.param_encoder.decode(nparam)

        print("shape of param: ", param.shape)

        test_before = True

        if test_before: 
            avg_reward_list, avg_success_list, avg_success_time_list = display_model(param, env)
            avg_reward = np.average(avg_reward_list)
            avg_success = np.average(avg_success_list)
            avg_success_time = np.average(avg_success_time_list)

            max_reward = np.max(avg_reward_list)
            min_time = np.min(avg_success_time_list)

        print("After diffusion generation.")

        gen_param = self.param_encoder.decode(pred_param)
        print("shape of generated param: ", gen_param.shape)
        gen_avg_reward_list, gen_avg_success_list, gen_avg_success_time_list = display_model(gen_param, env)
        gen_avg_reward = np.average(gen_avg_reward_list)
        gen_avg_success = np.average(gen_avg_success_list)
        gen_avg_success_time = np.average(gen_avg_success_time_list)

        gen_max_reward = np.max(gen_avg_reward_list)
        gen_min_time = np.min(gen_avg_success_time_list)


        gen_avg_reward_list.sort(reverse=True)
        gen_avg_success_list.sort(reverse=True)
        gen_avg_success_time_list.sort(reverse=False)

        gen_top_5_rewards = np.average(gen_avg_reward_list[:5])
        gen_top_10_rewards = np.average(gen_avg_reward_list[:10])
        gen_top_5_success = np.average(gen_avg_success_list[:5])
        gen_top_10_success = np.average(gen_avg_success_list[:10])
        gen_top_5_success_time = np.average(gen_avg_success_time_list[:5])
        gen_top_10_success_time = np.average(gen_avg_success_time_list[:10])

        if test_before: 
            print("Avg. Reward: {}, Avg. Success: {}, Avg Length: {}".format(round(avg_reward, 2), round(avg_success,2), round(avg_success_time,2)))
            print("Before generation, Max Reward: {}, Min Episode Length: {}".format(round(max_reward, 2), round(min_time,2)))

        print("After Generated, Avg. Reward: {}, Avg. Success: {}, Avg Length: {}".format(round(gen_avg_reward, 2), round(gen_avg_success,2), round(gen_avg_success_time,2)))
        print("After generation, Max Reward: {}, Min Episode Length: {}".format(round(gen_max_reward, 2), round(gen_min_time,2)))
        print("Generated Top 5 Reward: {}, Top 10 Reward: {}".format(round(gen_top_5_rewards, 2), round(gen_top_10_rewards,2)))
        print("Generated Top 5 Success Rate: {}, Top 10 Success Rate: {}".format(round(gen_top_5_success, 2), round(gen_top_10_success,2)))
        print("Generated Top 5 Success Time: {}, Top 10 Success Time: {}".format(round(gen_top_5_success_time, 2), round(gen_top_10_success_time,2)))



    def load_encoder(self, encoder_path, evaluate=True):
        print("Loading encoders from {}".format(encoder_path))
        encoder_ckpt = torch.load(encoder_path, map_location='cpu')
        weights_dict = {}
        weights = encoder_ckpt['state_dict']
        for k, v in weights.items():
            new_k = k.replace('model.', '') if 'model.' in k else k
            weights_dict[new_k] = v
        self.param_encoder.load_state_dict(weights_dict)
        self.param_encoder.eval()
    
    def load_checkpoint(self, ckpt_path, evaluate=True):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.normalizer.load_state_dict(checkpoint['normalizer'])

            if evaluate:
                self.model.eval()
            else:
                self.model.train()


@hydra.main(config_name="config")  

def main(cfg):
    workspace = EvalWorkspace(cfg)
    env_name = cfg.eval.env_name
    data_path = cfg.eval.data_dir
    data = torch.load(data_path)
    workspace.rollout(data, env=env_name)

if __name__ == "__main__":
    main()

