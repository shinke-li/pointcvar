from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.EXP_ID = ""
_C.EXP.SEED = 1
_C.EXP.TASK = 'cls'
_C.EXP.DATASET = 'modelnet40_dgcnn'
_C.EXP.MODEL_NAME = 'mv'
_C.EXP.LOSS_NAME = 'smooth'
_C.EXP.OPTIMIZER = 'vanilla'
_C.EXP.METRIC = 'acc'
#------------------------------------------------------------------------------
# Inference Experiment Parameters --cvar
#------------------------------------------------------------------------------
_C.ROBUSTINFER = CN()
_C.ROBUSTINFER.ACTIVATE = False 
_C.ROBUSTINFER.METHOD = 'cvar' #'dup'
_C.ROBUSTINFER.PARAMS = CN()
_C.ROBUSTINFER.PARAMS.CVAR = CN()
_C.ROBUSTINFER.PARAMS.CVAR.drop_rate = 100.
_C.ROBUSTINFER.PARAMS.CVAR.use_true = False 
_C.ROBUSTINFER.PARAMS.CVAR.drop_strategy = 'zero'
_C.ROBUSTINFER.PARAMS.CVAR.step = 1
_C.ROBUSTINFER.PARAMS.CVAR.iter_num = 1
_C.ROBUSTINFER.PARAMS.DUP = CN()
_C.ROBUSTINFER.PARAMS.DUP.model_path = "defense/DUP_Net/pu-in_1024-up_4.pth" 
_C.ROBUSTINFER.PARAMS.SOR = CN()
_C.ROBUSTINFER.PARAMS.SRS = CN()
_C.ROBUSTINFER.PARAMS.ROR = CN()
_C.ROBUSTINFER.PARAMS.PCN = CN()
_C.ROBUSTINFER.PARAMS.PCN.drop_rate = 0.1
_C.ROBUSTINFER.PARAMS.PCN.iter_num = 1
_C.ROBUSTINFER.PARAMS.DUP_CVAR = CN()
_C.ROBUSTINFER.PARAMS.DUP_CVAR.drop_rate = 100.
_C.ROBUSTINFER.PARAMS.DUP_CVAR.use_true = False 
_C.ROBUSTINFER.PARAMS.DUP_CVAR.drop_strategy = 'zero'
_C.ROBUSTINFER.PARAMS.DUP_CVAR.step = 1
_C.ROBUSTINFER.PARAMS.DUP_CVAR.iter_num = 1
_C.ROBUSTINFER.PARAMS.DUP_CVAR.model_path = "defense/DUP_Net/pu-in_1024-up_4.pth"

#_C.ROBUSTINFER.PARAMS.SOR.alpha = "defense/DUP_Net/pu-in_1024-up_4.pth" 
#_C.ROBUSTINFER.PARAMS.DUPNET = CN()
#_C.ROBUSTINFER.PARAMS.DUPNET.model_path = "" 

#------------------------------------------------------------------------------
# Extra Experiment Parameters
#------------------------------------------------------------------------------
_C.EXP_EXTRA = CN()
_C.EXP_EXTRA.no_val = True
_C.EXP_EXTRA.no_test = False
_C.EXP_EXTRA.val_eval_freq = 1
_C.EXP_EXTRA.test_eval_freq = 1
_C.EXP_EXTRA.save_ckp = 25
# -----------------------------------------------------------------------------
# DATALOADER (contains things common across the datasets)
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.batch_size = 32
_C.DATALOADER.num_workers = 0
# -----------------------------------------------------------------------------
# TRAINING DETAILS (contains things common across the training)
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.num_epochs = 250
_C.TRAIN.learning_rate = 1e-3
_C.TRAIN.lr_decay_factor = 0.5
_C.TRAIN.lr_reduce_patience = 10
_C.TRAIN.l2 = 0.0
_C.TRAIN.early_stop = 300
_C.TRAIN.lr_clip = 0.00001
#-----------------------------------------------------------------------------
# MODELNET40_DGCNN
#-----------------------------------------------------------------------------
_C.DATALOADER.MODELNET40_DGCNN = CN()
_C.DATALOADER.MODELNET40_DGCNN.train_data_path = './data/modelnet40_ply_hdf5_2048/train_files.txt' 
_C.DATALOADER.MODELNET40_DGCNN.valid_data_path = './data/modelnet40_ply_hdf5_2048/train_files.txt' 
_C.DATALOADER.MODELNET40_DGCNN.test_data_path  = './data/modelnet40_ply_hdf5_2048/test_files.txt'
_C.DATALOADER.MODELNET40_DGCNN.num_points      = 1024
#-----------------------------------------------------------------------------
# SHAPENET_PART
#-----------------------------------------------------------------------------
_C.DATALOADER.SHAPENET_PART = CN()
_C.DATALOADER.SHAPENET_PART.train_data_path = 'data/shapenetpart_hdf5_2048/train_files.txt'
_C.DATALOADER.SHAPENET_PART.valid_data_path = 'data/shapenetpart_hdf5_2048/train_files.txt'
_C.DATALOADER.SHAPENET_PART.test_data_path  = 'data/shapenetpart_hdf5_2048/test_files.txt'
_C.DATALOADER.SHAPENET_PART.num_points      = 1024
#-----------------------------------------------------------------------------
# MODELNET40_C
#-----------------------------------------------------------------------------
_C.DATALOADER.MODELNET40_C = CN()
_C.DATALOADER.MODELNET40_C.test_data_path  = './data/modelnet40_c/'
_C.DATALOADER.MODELNET40_C.corruption      = 'uniform'
_C.DATALOADER.MODELNET40_C.severity        = 1
# ----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# -----------------------------------------------------------------------------
# MV MODEL
# -----------------------------------------------------------------------------
_C.MODEL.MV = CN()
_C.MODEL.MV.backbone = 'resnet18'
_C.MODEL.MV.feat_size = 16
# -----------------------------------------------------------------------------
# RSCNN MODEL
# -----------------------------------------------------------------------------
_C.MODEL.RSCNN = CN()
_C.MODEL.RSCNN.ssn_or_msn = True
# -----------------------------------------------------------------------------
# PN2 MODEL
# -----------------------------------------------------------------------------
_C.MODEL.PN2 = CN()
_C.MODEL.PN2.version_cls = 1.0

_C.AUG = CN()
_C.AUG.NAME = 'none'
_C.AUG.BETA = 1.
_C.AUG.PROB = 0.5
_C.AUG.MIXUPRATE = 0.4

_C.ADAPT = CN()
_C.ADAPT.METHOD = 'none'
_C.ADAPT.ITER = 1


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
