from DVM import DVMTrainner,DVMR,DVDR
from mmengine import Registry,Config
import wandb

config_file_path = 'config/ventialtion_v1/trainner.py'

config = Config.fromfile(config_file_path)

trainner = DVMR.build(config.trainner)

wandb.login(key='b0678ed4e7952252ce531e0e26ee65271d81ae8f')
wandb.init(
    # set the wandb project where this run will be logged
    project=config.project_name,
    name='train_dvm',
    # track hyperparameters and run metadata
    config=config.wandb_config
)

trainner.train()