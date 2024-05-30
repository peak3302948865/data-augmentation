from yacs.config import CfgNode as CN
_C = CN()
import numpy as np
_C.RANDOM_SEED = 44
_C.LOG_PATH = 'logs/train.log'
# DATA config
_C.DATA = CN()
_C.DATA.name = 'VOC'
_C.DATA.dir = 'dataset/VOC2007'
_C.DATA.width = 416
_C.DATA.height = 416
_C.DATA.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
_C.DATA.num_classes = len(_C.DATA.classes)

# MODEL config
_C.MODEL = CN()
_C.MODEL.saved_path = 'trained_model/'

# train config
_C.TRAIN = CN()
_C.TRAIN.num_workers = 3
_C.TRAIN.optimizer = 'SGD'
_C.TRAIN.batch_size = 20
_C.TRAIN.lr = 1e-2
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.epochs = 20
_C.TRAIN.device = 'cuda:0'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()



