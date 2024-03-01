import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import os
from pc_utils import (rotate_point_cloud, PointcloudScaleAndTranslate)
from dgcnn.pytorch.data import ModelNet40 as dgcnn_ModelNet40
import random

BACKDOOR_TARGET = 30

class ModelNet40Dgcnn(Dataset):
    def __init__(self, split, train_data_path,
                 valid_data_path, test_data_path, num_points):
        self.split = split
        self.data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.split]

        dgcnn_params = {
            'partition': 'train' if split in ['train', 'valid'] else 'test',
            'num_points': num_points,
            "data_path":  self.data_path
        }
        self.dataset = dgcnn_ModelNet40(**dgcnn_params)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        pc, label = self.dataset.__getitem__(idx)
        return {'pc': pc, 'label': label.item()}

class ShapeNetPart(ModelNet40Dgcnn):
    def __init__(self, split, train_data_path, valid_data_path, test_data_path, num_points):
        super().__init__(split, train_data_path, valid_data_path, test_data_path, num_points)
    def __len__(self):
        return self.dataset.__len__()  
    def __getitem__(self, idx):
        pc, label = self.dataset.__getitem__(idx)
        return {'pc': pc, 'label': label.item()}


def load_data(data_path,corruption,severity, target=False):
    DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    if (corruption in ['uniform', 'gaussian', 'background','impulse','upsampling','distortion_rbf','distortion_rbf_inv','density', 'density_inc','shear','rotation','cutout','distortion' ,'occlusion','lidar']):
        LABEL_DIR = os.path.join(data_path, 'label.npy') 
    else:
        if not target:
            LABEL_DIR = os.path.join(data_path, 'label_'+ corruption + '_' +str(severity) + '.npy')
        else:
            LABEL_DIR = os.path.join(data_path,'target_label_'+ corruption + '_' +str(severity) + '.npy')
            if 'backdoor' in corruption:
                true_label = np.load(os.path.join(data_path, 'label_'+ corruption + '_' +str(severity) + '.npy'))
                false_label = np.full_like(true_label, BACKDOOR_TARGET)
                false_label[true_label==BACKDOOR_TARGET] = 255
                np.save(LABEL_DIR, false_label)
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)
    return all_data, all_label

class ModelNet40C(Dataset):
    def __init__(self, split, test_data_path,corruption,severity, target=False):
        assert split == 'test'
        self.split = split
        self.data_path = {
            "test":  test_data_path
        }[self.split]
        self.corruption = corruption
        self.severity = severity
        self.data, self.label = load_data(self.data_path, self.corruption, self.severity, target)
        self.partition =  'test'

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return {'pc': pointcloud, 'label': label.item()}

    def __len__(self):
        return self.data.shape[0]


def create_dataloader(split, cfg):
    num_workers = cfg.DATALOADER.num_workers
    batch_size = cfg.DATALOADER.batch_size
    dataset_args = {
        "split": split,
    }

    if cfg.EXP.DATASET == "modelnet40_dgcnn":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_DGCNN))
        dataset = ModelNet40Dgcnn(**dataset_args)
    elif cfg.EXP.DATASET == "shapenetpart_dgcnn":  
        dataset_args.update(dict(**cfg.DATALOADER.SHAPENET_PART))
        dataset = ShapeNetPart(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_c":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_C))
        dataset = ModelNet40C(**dataset_args)
    elif cfg.EXP.DATASET == "shapenet_c":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_C))
        dataset = ModelNet40C(**dataset_args)
    else:
        assert False

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = None

    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        pin_memory=(torch.cuda.is_available()) and (not num_workers)
    )


def create_target_dataloader(split, cfg):
    num_workers = cfg.DATALOADER.num_workers
    batch_size = cfg.DATALOADER.batch_size
    dataset_args = {
        "split": split
    }
    if cfg.EXP.DATASET == "modelnet40_dgcnn":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_DGCNN))
        dataset = ModelNet40Dgcnn(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_c":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_C))
        dataset = ModelNet40C(target=True, **dataset_args)
    else:
        assert False

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = None

    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        pin_memory=(torch.cuda.is_available()) and (not num_workers)
    )
