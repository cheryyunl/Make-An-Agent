# Make-An-Agent: A Generalizable Policy Network Generator with Behavior-Prompted Diffusion
<p align="center" style="font-size: 50px">
   <a href="">[Paper]</a>&emsp;<a href="https://cheryyunl.github.io/make-an-agent/">[Project Website]</a>
</p>

This repository is the official PyTorch implementation of **Make-An-Agent**. **Make-An-Agent**, a policy parameter generator that leverages the power of conditional diffusion models for behavior-to-policy generation, which demonstrates remarkable versatility and scalability on multiple tasks and has a strong generalization ability on unseen tasks to output well-performed policies with only few-shot demonstrations as inputs. 

<p align="center">
  <br><img src='images/teaser.gif' width="700"/><br>
</p>

# 💻 Installation
1. create a virtual environment and install all required packages. 
```bash
conda env create -f environment.yml 
conda activate makeagent
```

2. install Metaworld and mujoco_py for evaluations.
Following instructions in [DrM](https://github.com/XuGW-Kevin/DrM).

## 🛠️ Code Usage

For training autoencoder and behavior embedding, you could download the training dataset from [Huggingface](https://huggingface.co/cheryyunl/Make-An-Agent): `train_data/training_dataset.pt` or specific data for each task, e.g. `train_data/door-open.pt`.

Training the parameter autoencoder to encode and decode policy network parameters:
Change `data_root` in `autoencoder/config.yaml`.
```bash
cd autoencoder
python train.py
```

Training behavior embeddings to process trajectory data:
Change `data_root` in `behavior_embedding/config_embed.yaml`.
```bash
cd behavior_embedding
python train.py
```

Training the policy generator with conditional diffusion models:

Data processing: 
Make-An-Agent uses latent diffusion model, so the data should be processed using the autoencoder and behavior embedding.
You can direclty use the pretrained models in [HuggingFace](https://huggingface.co/cheryyunl/Make-An-Agent)

Or directly use the processed training data in `train_data/process_data.pt` to train the policy generator.

If you want to process your own data, change the paths of data and pretrained model root in `dataset/config.yaml`.
```bash
cd dataset.py
python process_data.py
```

Ensure you now have processed data to match the latent representation dimensions, then change `data_root` in `PolicyGenerator/config.yaml` with your processed data. 
```bash
cd PolicyGenerator
python train.py
```


Evaluating the synthesized parameters:

Change `data_root` in `PolicyGenerator/config.yaml`.
```bash
cd PolicyGenerator/policy_generator
python eval.py
```

## 📗 Dataset and Pretrained Models


## 📝 Citation

If you find our work or code useful, please consider citing as follows:

```
@article{liang2024make,
title={Make-An-Agent: A Generalizable Policy Network Generator with Behavior-Prompted Diffusion},
author={Liang, Yongyuan and Xu, Tingqiang and Hu, Kaizhe and Jiang, Guangqi and Huang, Furong and Xu, Huazhe},
journal={arXiv preprint arXiv:2407.10973},
year={2024}
}
```

## 🌷 Acknowledgement
Our work is primarily based on the following projects: [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning), [Metaworld](https://github.com/Farama-Foundation/Metaworld), [Robosuite](https://github.com/ARISE-Initiative/robosuite), [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways). We thank these authors for their contributions to the open-source community.
For any questions or suggestions, please contact [Yongyuan Liang](https://cheryyunl.github.io/).
