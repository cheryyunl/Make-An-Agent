import math
import pdb
import random

import torch
from torch import nn
import torch.nn.functional as F

def build_encoder(n_embd, encoder_depth, input_splits):
    # Create a unique MLP encoder for each token
    input_parameter_projections = nn.ModuleList()
    for param_chunk_size in input_splits:
        in_proj = [nn.Linear(param_chunk_size, n_embd, bias=False)]
        for _ in range(encoder_depth - 1):
            in_proj.append(nn.GELU())
            in_proj.append(nn.Linear(n_embd, n_embd, bias=False))
        in_proj = nn.Sequential(*in_proj)
        input_parameter_projections.append(in_proj)
    return input_parameter_projections

def build_decoder(n_embd, decoder_depth, output_splits):
    # Create a unique MLP decoder for each noised token
    output_parameter_projections = nn.ModuleList()
    for output_chunk_size in output_splits:
        out_proj = []
        for _ in range(decoder_depth - 1):
            out_proj.append(nn.Linear(n_embd, n_embd, bias=False))
            out_proj.append(nn.GELU())
        out_proj.append(nn.Linear(n_embd, output_chunk_size, bias=False))
        out_proj = nn.Sequential(*out_proj)
        output_parameter_projections.append(out_proj)
    return output_parameter_projections


class EncoderDecoder(nn.Module):
    def __init__(self, n_embd, encoder_depth=1, decoder_depth=1, input_noise_factor=0.0001, latent_noise_factor=0.001):
        super().__init__()
        self.input_splits = [5120, 17544]
        self.input_parameter_projections = build_encoder(n_embd, encoder_depth, self.input_splits)
        self.output_parameter_projections = build_decoder(n_embd, decoder_depth, self.input_splits)
        self.num_output_heads = len(self.input_splits)
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor
        self.ln_in = ln_in = nn.LayerNorm(n_embd)

    def encode(self, parameters):
        """
        Chunk input parameter vector, apply per-chunk encoding, and
        stack projected chunks along the sequence (token) dimension.
        """
        assert parameters.dim() == 2
        split_parameters = torch.split(parameters, self.input_splits, dim=1)
        representations = []
        for parameter, in_proj in zip(split_parameters, self.input_parameter_projections):
            representations.append(in_proj(parameter))
        representations = torch.stack(representations, dim=1)  
        representations = self.ln_in(representations)
        assert representations.dim() == 3
        return representations

    def decode(self, features):
        """
        Apply a per-chunk decoding (only to the tokens corresponding to the noised/updated parameter vector),
        and concatenate them into a flattened parameter vector.
        """
        assert features.dim() == 3  # (b, t, d)
        output = []
        for t in range(self.num_output_heads):
            out_proj = self.output_parameter_projections[t]
            output.append(out_proj(features[:, t, :]))
        output = torch.cat(output, 1)  # (b, c)
        assert output.dim() == 2
        return output

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

    def forward(self, x):
        x = self.add_noise(x, self.input_noise_factor)
        x = self.encode(x)
        x = self.add_noise(x, self.latent_noise_factor)
        x = torch.clamp(x, -1, 1)
        x = self.decode(x)
        return x

