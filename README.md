# Make-An-Agent: A Generalizable Policy Network Generator with Behavior-Prompted Diffusion
<p align="center" style="font-size: 50px">
   <a href="">[Paper]</a>&emsp;<a href="https://cheryyunl.github.io/make-an-agent/">[Project Website]</a>
</p>

This repository is the official PyTorch implementation of **Make-An-Agent**. **Make-An-Agent**, a policy parameter generator that leverages the power of conditional diffusion models for behavior-to-policy generation, which demonstrates remarkable versatility and scalability on multiple tasks and has a strong generalization ability on unseen tasks to output well-performed policies with only few-shot demonstrations as inputs. 

<p align="center">
  <br><img src='images/' width="500"/><br>
</p>

# 💻 Installation
1. create a virtual environment and install all required packages. 
```bash
conda env create -f environment.yml 
conda activate makeagent
```

2. install Metaworld and mujoco_py for evaluations.
```bash
conda env create -f environment.yml 
conda activate makeagent
```


## 🛠️ Code Usage
If you would like to run DrM on [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), please use train_dmc.py to train DrM policies on different configs.

```bash
python train_dmc.py task=dog_walk agent=drm
```

If you would like to run DrM on [MetaWorld](https://meta-world.github.io/), please use train_mw.py to train DrM policies on different configs.

```bash
python train_mw.py task=coffee-push agent=drm_mw
python train_mw.py task=disassemble agent=drm_mw
```

If you would like to run DrM on Adroit, please use train_adroit.py to train DrM policies on different configs.

```bash
python train_adroit.py task=pen agent=drm_adroit
```

## 📝 Citation

If you find our work or code useful, please consider citing as follows:

```
\cite
```

Our work is primarily based on the following projects: [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning), [Metaworld](https://github.com/Farama-Foundation/Metaworld), [Robosuite](https://github.com/ARISE-Initiative/robosuite), [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways). We thank these authors for their contributions to the open-source community.
For any questions or suggestions, please contact [Yongyuan Liang](https://cheryyunl.github.io/).

## 🌷 Acknowledgement
DrM is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. We would like to thank DrQ-v2 authors for open-sourcing the [DrQv2](https://github.com/facebookresearch/drqv2) codebase. Our implementation builds on top of their repository.
