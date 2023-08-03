from mmengine import Registry
from mmengine import MODELS
from mmengine import DATASETS

DVM = Registry('DVM', parent=MODELS)
DVD = Registry('DVD', parent=DATASETS)
