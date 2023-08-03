from DVM.registry import DVM
from mmengine import Config
import wandb
import torch
config = Config.fromfile('config/ventialtion_v1/trainner.py')
wandb.login(key='b0678ed4e7952252ce531e0e26ee65271d81ae8f')
wandb.init(
    # set the wandb project where this run will be logged
    project=config.project_name,
    name='trainning_ldm',
    # track hyperparameters and run metadata
    config=config.wandb_config
)

ldm_trainner = DVM.build(config.ldm_trainner)

ldm_trainner.train()