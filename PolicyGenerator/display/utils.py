import torch

def param_to_policy(flattened_params, policy):
    layer_idx = 0
    for name, param in policy.items():
        param_shape = param.shape
        param_length = param.view(-1).shape[0]
        param.data = flattened_params[layer_idx:layer_idx + param_length].reshape(param_shape)
        param.data.to(flattened_params.device)
        layer_idx += param_length
    return policy

def policy_to_param(policy):
    param = []
    for key in policy.keys():
        param.append(policy[key].data.reshape(-1))
    param = torch.cat(param, 0)
    return param
