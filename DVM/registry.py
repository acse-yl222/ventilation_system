from mmengine import Registry
from mmengine import MODELS
from mmengine import DATASETS

DVMR = Registry('DVMR', parent=MODELS)
DVDR = Registry('DVDR', parent=DATASETS)
