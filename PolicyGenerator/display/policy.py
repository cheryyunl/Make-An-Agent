import os
import numpy as np
import ipdb
import torch
import copy
import torch.nn.functional as F
from torch.optim import Adam
from display.model import GaussianPolicy

class SAC(object):
    def __init__(self, num_inputs, action_space, device):

        self.device = device

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], 128, action_space)
        self.policy.eval()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
