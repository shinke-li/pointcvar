"""Untargeted salient point dropping attack."""

import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('../')

from config import BEST_WEIGHTS
from config import MAX_DROP_BATCH as BATCH_SIZE
from dataset import ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, Args, Pct
from util.utils import str2bool, set_seed
from attack import SaliencyDrop
from collections import OrderedDict


def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    num = 0
    for pc, label, target in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(), label.long().cuda()
            target_label = target.long().cuda()

        # input GT label here because it's untargeted attack
        best_pc, success_num = attacker.attack(pc, label)

        # results
        num += success_num
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())
        all_target_lbl.append(target_label.detach().cpu().numpy())

    # accumulate results
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(all_target_lbl, axis=0)  # [num_data]
    return all_adv_pc, all_real_lbl, all_target_lbl, num


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='data/attack_data.npz')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pct','pgd_pointnet', 'cutmix_r_pointnet','cutmix_k_dgcnn','cutmix_r_dgcnn',
                                 'pgd_dgcnn', 'cutmix_k_pointnet','dgcnn','pointnet','cvar_random_pointnet','cvar_drop_pointnet',
                                 'cvar_zero_pointnet', 'cvar_expand_pointnet','cvar_randomexpand_pointnet','cvar_random_dgcnn',
                                 'cvar_drop_dgcnn','cvar_zero_dgcnn','cvar_expand_dgcnn','cvar_randomexpand_dgcnn','cvar_randomexpand_pointnet_droprate6_step1',
                                 'cvar_randomexpand_pointnet_2.0_droprate11_step1_run_1'],)
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40', 'opt_mn40', 'conv_opt_mn40'])
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--num_drop', type=int, default=200, metavar='N',
                        help='Number of dropping points')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_points]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_points]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    cudnn.benchmark = True

    # build model
    args1=Args()
    if args.model.lower() == 'cutmix_k_dgcnn':
        model = DGCNN(args1, output_channels=40)
    elif args.model.lower() == 'dgcnn':
        model = DGCNN(args1, output_channels=40)
    elif args.model.lower() == 'pct':
        model = Pct(args1, output_channels=40)
    elif args.model.lower() == 'cutmix_r_dgcnn':
        model = DGCNN(args1, output_channels=40)
    elif args.model.lower() == 'pgd_dgcnn':
        model = DGCNN(args1, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'cutmix_k_pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'cutmix_r_pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pgd_pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    
    model = nn.DataParallel(model).cuda()

    # load model weight
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    state_dict = torch.load(BEST_WEIGHTS[args.model])
    state_dict = OrderedDict([(k.replace('model.',''), v) for k, v in state_dict['model_state'].items()])
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.module.load_state_dict(state_dict)

    # setup attack settings
    # hyper-parameters from their official tensorflow code
    attacker = SaliencyDrop(model, num_drop=args.num_drop,
                            alpha=1, k=5)

    # attack
    test_set = ModelNet40Attack(args.data_root, num_points=args.num_points,
                                normalize=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False)

    # run attack
    attacked_data, real_label, target_label, correct_num = attack()

    # accumulate results
    data_num = len(test_set)
    acc = float(correct_num) / float(data_num)

    # save results
    save_path = './attack/results/{}_{}/Drop_{}'.\
        format(args.dataset, args.num_points, args.num_drop)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = 'Drop-{}-acc_{:.4f}.npz'.\
        format(args.model, acc)
    np.savez(os.path.join(save_path, save_name),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8),
             target_label=target_label.astype(np.uint8))
