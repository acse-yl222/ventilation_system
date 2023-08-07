from DVM import DVMTrainner,DVMR,DVDR
from mmengine import Registry,Config
import wandb

config_file_path = 'config/ventialtion_v1/evaler.py'

config = Config.fromfile(config_file_path)

evaler = DVMR.build(config.evaler)

evaler.eval()