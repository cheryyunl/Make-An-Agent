import random

import torch
from torch import nn
import torch.nn.functional as F
from normalizer import LinearNormalizer
from dataset import Dataset

class Embedding(nn.Module):
    def __init__(self, cfg):
        super(Embedding, self).__init__()
        self.traj_dim = cfg.traj_dim
        self.task_dim = cfg.task_dim
        self.hidden_size = cfg.hidden_size
        self.feature_dim = cfg.feature_dim

        self.traj_embed = nn.Sequential(
                            nn.Linear(self.traj_dim, self.hidden_size),
                            nn.ReLU(),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.ReLU(),
                            nn.Linear(self.hidden_size, self.feature_dim))
        
        self.task_embed = nn.Sequential(nn.Linear(self.task_dim, 256),
                                        nn.LayerNorm(256), nn.Tanh(),
                                        nn.Linear(256, self.feature_dim))
        
        self.W = nn.Parameter(torch.rand(self.feature_dim, self.feature_dim))
    
    def compute_logits(self, traj_e, task_e):
        """
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        
        Wz = torch.matmul(self.W, task_e.T)  # (z_dim,B)
        logits = torch.matmul(traj_e, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
    

    def traj_embedding(self, x):
        x = x.to(torch.float32)
        return self.traj_embed(x)
    
    def task_embedding(self, x):
        x = x.to(torch.float32)
        return self.task_embed(x)
    
    def forward(self, traj, task):
        traj = traj.to(torch.float32)
        task = task.to(torch.float32)
        traj_e = self.traj_embedding(traj)
        task_e = self.task_embedding(task)
        return traj_e, task_e





        
