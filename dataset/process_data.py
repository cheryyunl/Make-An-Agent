import os
import sys,os
import random
import torch
from autoencoder.model_proj import EncoderDecoder
from behavior_embedding.embed_model import Embedding
import hydra

def process_weights(state_dict):
    weights = {}
    for k, v in state_dict.items():
        if 'model.' in k: 
            new_k = k.replace('model.', '')
            weights[new_k] = v
    return weights


@hydra.main(config_name="config")  
def main(cfg):
    
    root_dir = cfg.dir.root_dir
    data_dir = root_dir
    processed = root_dir + 'processed/'

    encoder_dir = cfg.dir.encoder_dir
    embedding_dir = cfg.dir.embed_dir
    encoder_ckpt = torch.load(encoder_dir, map_location = 'cpu')
    embedding_ckpt = torch.load(embedding_dir, map_location = 'cpu')

    for filename in os.listdir(data_dir):
        if filename.endswith(".pt"):
            data = torch.load(data_dir + filename, map_location = 'cpu')
            processed_dir = processed + filename

            process_data = {
                "param": [],
                "traj": [],
                "task": []
            }

            param_encoder = EncoderDecoder(1024, 1, 1, 0.0001, 0.001)

            behavior_embedding = Embedding(cfg.embedding_model)

            param_encoder.load_state_dict(process_weights(encoder_ckpt['state_dict']))
            behavior_embedding.load_state_dict(process_weights(embedding_ckpt['state_dict']))

            encode_param = param_encoder.encode(data["param"]).detach()
            embed_traj = behavior_embedding.traj_embedding(data["traj"]).detach()
            embed_task = behavior_embedding.task_embedding(data["task"]).detach()

            process_data["param"] = encode_param
            process_data["traj"] = embed_traj
            process_data['task'] = embed_task

            torch.save(process_data, processed_dir)
            print("param:", process_data["param"].shape)
            print("trajectory:", process_data["traj"].shape)
            print("task:", process_data["task"].shape)


if __name__ == "__main__":
    main()






