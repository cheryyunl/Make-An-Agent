# Make-An-Agent: A Generalizable Policy Network Generator with Behavior-Prompted Diffusion
<p align="center" style="font-size: 50px">
   <a href="">[Paper]</a>&emsp;<a href="https://cheryyunl.github.io/make-an-agent/">[Project Website]</a>
</p>

This repository is the official PyTorch implementation of **Make-An-Agent**. **Make-An-Agent**, a policy parameter generator that leverages the power of conditional diffusion models for behavior-to-policy generation, which demonstrates remarkable versatility and scalability on multiple tasks and has a strong generalization ability on unseen tasks to output well-performed policies with only few-shot demonstrations as inputs. 

<p align="center">
  <br><img src='images/teaser.gif' width="500"/><br>
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

Training the parameter autoencoder to encode and decode policy network parameters:
```bash
cd encoder
python train.py
```

Training behavior embeddings to process trajectory data:

```bash
python train_mw.py task=coffee-push agent=drm_mw
python train_mw.py task=disassemble agent=drm_mw
```

Training the policy generator with conditional diffusion models:

```bash
cd PolicyGenerator/policy_generator
python train.py
```

Evaluating the synthesized parameters:
```bash
cd PolicyGenerator/policy_generator
python eval.py
```

## 📝 Citation

If you find our work or code useful, please consider citing as follows:

```
\cite
```

## 🌷 Acknowledgement
Our work is primarily based on the following projects: [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning), [Metaworld](https://github.com/Farama-Foundation/Metaworld), [Robosuite](https://github.com/ARISE-Initiative/robosuite), [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways). We thank these authors for their contributions to the open-source community.
For any questions or suggestions, please contact [Yongyuan Liang](https://cheryyunl.github.io/).
