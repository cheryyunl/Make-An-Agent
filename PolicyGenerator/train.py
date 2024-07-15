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
from policy_generator.policy.policy_generator import PolicyGenerator
from policy_generator.model.common.lr_scheduler import get_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from policy_generator.model.diffusion.ema_model import EMAModel
from policy_generator.common.checkpoint_util import TopKCheckpointManager
from policy_generator.common.pytorch_util import dict_apply, optimizer_to


class Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

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

        self.ema_model: PolicyGenerator  = None
        if cfg.train.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.best_score = 10
        self.val_score = 0

        self.output_dir = '/path/outputs/generator-{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M"))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.train.debug:
            cfg.train.num_epochs = 100
            cfg.train.max_train_steps = 10
            cfg.train.max_val_steps = 3
            cfg.train.checkpoint_every = 1
            cfg.train.val_every = 1
            cfg.train.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = True
        
        # resume training
        if cfg.train.resume:
            lastest_ckpt_path = self.output_dir
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset = Dataset(cfg.data) 
        normalizer = dataset.get_normalizer()
        train_dataloader = dataset.train_dataloader()
        val_dataloader = dataset.val_dataloader()
        self.normalizer = normalizer
        
        self.model.set_normalizer(normalizer)
        if cfg.train.use_ema:
            self.ema_model.set_normalizer(normalizer)
        
        lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-9)

        # configure ema
        ema: EMAModel = None
        if cfg.train.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        cfg.logging.name = str(cfg.logging.name)
        print("-----------------------------")
        print(f"[WandB] group: {cfg.logging.group}")
        print(f"[WandB] name: {cfg.logging.name}")
        print("-----------------------------")

        experiment_name = cfg.logging.name
        run_id = '{}-{}-{}-{}-{}-{}'.format(cfg.policy.kernel_size, cfg.optimizer.lr, cfg.data.batch_size, cfg.policy.use_traj, cfg.policy.use_task, datetime.datetime.now().strftime("%Y-%m-%d_%H_%M"))

        wandb_run = wandb.init(
        project = str(cfg.logging.project),
        config = {
            "seed": cfg.train.seed,
            "batch_size": cfg.data.batch_size,
            "lr": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
            "eps": cfg.optimizer.eps,
            "num_epochs": cfg.train.num_epochs,
            "num_inference_steps": cfg.policy.num_inference_steps,
            "kernel_size": cfg.policy.kernel_size,
            "use_traj": cfg.policy.use_traj,
            "use_task": cfg.policy.use_task
        },
        name = experiment_name,
        id = run_id,
        save_code = False
        )

        # device transfer
        device = torch.device(cfg.train.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        # save batch for sampling
        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.train.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.train.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()

                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.train.gradient_accumulate_every
                    loss.backward()

                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.train.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.train.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.train.max_train_steps is not None) \
                        and batch_idx >= (cfg.train.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            generator = self.model
            if cfg.train.use_ema:
                generator = self.ema_model
            generator.eval()

            # run validation
            if (self.epoch % cfg.train.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.train.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.train.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss
                        self.val_score = val_loss
            
            # test_traj = batch['traj']
            # test_result = self.predict(test_traj)
            # print(test_result['param'].shape)
            # run diffusion sampling on a training batch
            # pass

            # checkpoint
            if (self.epoch % cfg.train.checkpoint_every) == 0:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint(tag='regular')
                if cfg.checkpoint.save_best_ckpt:
                    self.save_checkpoint(tag='best')
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

            generator.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

        self.save_checkpoint(tag = 'last')
        wandb.finish()

    def save_checkpoint(self, tag = 'regular'):
        if tag == 'regular':
            ckpt_path = self.output_dir + '/' + '{}-{}.torch'.format(self.epoch, self.val_score)
            torch.save({'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'normalizer': self.normalizer.state_dict()}, ckpt_path)
        elif tag == 'best':
            if self.val_score < self.best_score:
                self.best_score = self.val_score
                ckpt_path = self.output_dir  + '/' + 'best.torch'
                print(f'Saving the best model with {self.val_score} on {self.epoch}')
                torch.save({'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'normalizer': self.normalizer.state_dict()}, ckpt_path)
        elif tag == 'last':
            print(f'Saving the last model with {self.val_score} on {self.epoch}')
            ckpt_path = self.output_dir  + '/' + 'last-model.torch'
            torch.save({'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'normalizer': self.normalizer.state_dict()}, ckpt_path)

    def load_checkpoint(self, evaluate=False):
        ckpt_path = self.output_dir + '/' + 'best.torch'
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

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)

@hydra.main(config_name="config")  

def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

