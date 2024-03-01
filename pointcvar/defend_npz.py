"""Apply baseline defense methods"""
import os
import tqdm
import argparse
import numpy as np

import torch

from defense import SRSDefense, SORDefense, DUPNet,RORDefense
PU_NET_WEIGHT = 'defense/DUP_Net/pu-in_1024-up_4.pth'

def defend(data_batch, one_defense,args, use_sor=True, **kwargs):
    # save defense result
    '''
    sub_roots = data_root.split('/')
    filename = sub_roots[-1]
    data_folder = data_root[:data_root.rindex(filename)]
    save_folder = os.path.join(data_folder, one_defense)
    save_name = '{}_{}'.format(one_defense, filename)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    '''
    # data to defend
    batch_size = 128
    test_pc = data_batch['pc'].numpy()
    test_label = data_batch['label']
    #target_label = npz_data['target_label']

    # defense module
    if one_defense == 'ror':
        defense_module = RORDefense(n=args.ror_nb, r=args.ror_radius)
    elif one_defense == 'srs':
        defense_module = SRSDefense(drop_num=args.srs_drop_num)
    elif one_defense == 'sor':
        defense_module = SORDefense(k=args.sor_k, alpha=args.sor_alpha)
    elif one_defense == 'dup':
        up_ratio = 4
        defense_module = DUPNet(sor_k=args.sor_k,
                                sor_alpha=args.sor_alpha,
                                npoint=1024, 
                                up_ratio=up_ratio, 
                                use_sor=use_sor)
        defense_module.pu_net.load_state_dict(
            torch.load(PU_NET_WEIGHT))
        defense_module.pu_net = defense_module.pu_net.cuda()

    # defend
    all_defend_pc = []
    for batch_idx in range(0, len(test_pc), batch_size): #tqdm.trange
        batch_pc = test_pc[batch_idx:batch_idx + batch_size]
        batch_pc = torch.from_numpy(batch_pc)[..., :3]
        batch_pc = batch_pc.float().cuda()
        defend_batch_pc = defense_module(batch_pc)

        # sor processed results have different number of points in each
        if isinstance(defend_batch_pc, list) or \
                isinstance(defend_batch_pc, tuple):
            sor_defend_batch_pc = [
                pc.detach().cpu().numpy().astype(np.float32) for
                pc in defend_batch_pc
            ]
            _max_size = np.max([pc.shape[0] for pc in sor_defend_batch_pc])
            defend_batch_pc = []
            for pc in sor_defend_batch_pc:
                new_pc = np.zeros((_max_size, 3))
                new_pc[:pc.shape[0],:] = pc
                defend_batch_pc.append(new_pc)
        else:
            defend_batch_pc = defend_batch_pc.\
                detach().cpu().numpy().astype(np.float32)
            defend_batch_pc = [pc for pc in defend_batch_pc]

        all_defend_pc += defend_batch_pc

    all_defend_pc = np.array(all_defend_pc)
    if isinstance(all_defend_pc, np.ndarray): all_defend_pc = all_defend_pc.astype(np.float32)
    data_batch['pc'] = torch.tensor(all_defend_pc)
    data_batch['label'] = test_label
    return data_batch