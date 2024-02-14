import numpy as np
import torch
import os 
import h5py 
import torch.nn.functional as F
import sys
from main import get_loss
sys.path.append("./emd/")
import emd_module as emd
from operator import attrgetter
from functools import partial
from cvar_utils import get_cvar
import torch_scatter
from local_risk import calculate_local_risk

RETURN_RISK=False
CACULATE_CVAR=True

def cutmix_r(data_batch,cfg):
    r = np.random.rand(1)
    if cfg.AUG.BETA > 0 and r < cfg.AUG.PROB:
        lam = np.random.beta(cfg.AUG.BETA, cfg.AUG.BETA)
        B = data_batch['pc'].size()[0]
        rand_index = torch.randperm(B).cuda()
        target_a = data_batch['label']
        data_batch['label'] = data_batch['label'].to(rand_index.device)
        target_b = data_batch['label'][rand_index]

        point_a = torch.zeros(B, 1024, 3)
        point_b = torch.zeros(B, 1024, 3)
        point_c = torch.zeros(B, 1024, 3)
        data_batch['pc'] = data_batch['pc'].to(rand_index.device)
        point_a = data_batch['pc']
        point_b = data_batch['pc'][rand_index]
        point_c = data_batch['pc'][rand_index]

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(point_a, point_b, 0.005, 300)
        for ass in range(B):
            point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

        int_lam = int(cfg.DATALOADER.MODELNET40_DGCNN.num_points * lam)
        int_lam = max(1, int_lam)
        gamma = np.random.choice(cfg.DATALOADER.MODELNET40_DGCNN.num_points, int_lam, replace=False, p=None)
        for i2 in range(B):
            data_batch['pc'][i2, gamma, :] = point_c[i2, gamma, :]

        lam = int_lam * 1.0 / cfg.DATALOADER.MODELNET40_DGCNN.num_points
        # points = data_batch['pc'].transpose(2, 1)
        data_batch['label_2'] = target_b
        data_batch['lam'] = lam

    return data_batch
    


def cutmix_k(data_batch,cfg):
    r = np.random.rand(1)
    if cfg.AUG.BETA > 0 and r < cfg.AUG.PROB:
        lam = np.random.beta(cfg.AUG.BETA, cfg.AUG.BETA)
        B = data_batch['pc'].size()[0]

        rand_index = torch.randperm(B).cuda()
        target_a = data_batch['label']
        data_batch['label'] = data_batch['label'].to(rand_index.device)
        target_b = data_batch['label'][rand_index]

        point_a = torch.zeros(B, 1024, 3)
        point_b = torch.zeros(B, 1024, 3)
        point_c = torch.zeros(B, 1024, 3)
        point_a = data_batch['pc']
        data_batch['pc'] = data_batch['pc'].to(rand_index.device)
        point_b = data_batch['pc'][rand_index]
        point_c = data_batch['pc'][rand_index]

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(point_a, point_b, 0.005, 300)
        for ass in range(B):
            point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

        int_lam = int(cfg.DATALOADER.MODELNET40_DGCNN.num_points * lam)
        int_lam = max(1, int_lam)

        random_point = torch.from_numpy(np.random.choice(1024, B, replace=False, p=None))
        ind1 = torch.tensor(range(B))
        query = point_a[ind1, random_point].view(B, 1, 3)
        dist = torch.sqrt(torch.sum((point_a - query.repeat(1, cfg.DATALOADER.MODELNET40_DGCNN.num_points, 1)) ** 2, 2))
        idxs = dist.topk(int_lam, dim=1, largest=False, sorted=True).indices
        for i2 in range(B):
            data_batch['pc'][i2, idxs[i2], :] = point_c[i2, idxs[i2], :]
        lam = int_lam * 1.0 / cfg.DATALOADER.MODELNET40_DGCNN.num_points
        data_batch['label_2'] = target_b
        data_batch['lam'] = lam
        
    return data_batch


def mixup(data_batch,cfg):

    batch_size = data_batch['pc'].size()[0]
    idx_minor = torch.randperm(batch_size)
    mixrates = (0.5 - np.abs(np.random.beta(cfg.AUG.MIXUPRATE, cfg.AUG.MIXUPRATE, batch_size) - 0.5))
    label_main = data_batch['label']
    label_minor = data_batch['label'][idx_minor]
    label_new = torch.zeros(batch_size, 40)
    for i in range(batch_size):
        if label_main[i] == label_minor[i]: # same label
            label_new[i][label_main[i]] = 1.0
        else:
            label_new[i][label_main[i]] = 1 - mixrates[i]
            label_new[i][label_minor[i]] = mixrates[i]
    label = label_new

    data_minor = data_batch['pc'][idx_minor]
    mix_rate = torch.tensor(mixrates).float()
    mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)

    mix_rate_expand_xyz = mix_rate.expand(data_batch['pc'].shape)

    remd = emd.emdModule()
    remd = remd.cuda()
    _, ass = remd(data_batch['pc'], data_minor, 0.005, 300)
    ass = ass.long()
    data_minor = data_minor.to(ass.device) 
    data_batch['pc'] = data_batch['pc'].to(ass.device)
    mix_rate_expand_xyz = mix_rate_expand_xyz.to(ass.device) 

    for i in range(batch_size):
        
        data_minor[i] = data_minor[i][ass[i]]
    data_batch['pc'] = data_batch['pc'] * (1 - mix_rate_expand_xyz) + data_minor * mix_rate_expand_xyz
    data_batch['label_2'] = label_minor
    data_batch['lam'] = torch.tensor(mix_rate).squeeze_()

    return data_batch


def knn_points(k, xyz, query, nsample=512):
    B, N, C = xyz.shape
    _, S, _ = query.shape # S=1
    
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    sqrdists = square_distance(query, xyz) 
    tmp = np.sort(sqrdists, axis=2)
    knn_dist = np.zeros((B,1))
    for i in range(B):
        knn_dist[i][0] = tmp[i][0][k]
        group_idx[i][sqrdists[i]>knn_dist[i][0]]=N
    # group_idx[sqrdists > radius ** 2] = N
    # print("group idx : \n",group_idx)
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
    
def cut_points_knn(data_batch, idx, radius, nsample=512, k=512):

    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = knn_points(k=k, xyz=data_batch[:,:,:3], query=query_points[:,:,:3], nsample=nsample)
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6

def cut_points(data_batch, idx, radius, nsample=512):

    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = query_ball_point_for_rsmix(radius, nsample, data_batch[:,:,:3], query_points[:,:,:3])
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6


def query_ball_point_for_rsmix(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample], S=1
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * np.matmul(src, dst.transpose(0, 2, 1))
    dist += np.sum(src ** 2, -1).reshape(B, N, 1)
    dist += np.sum(dst ** 2, -1).reshape(B, 1, M)
    
    return dist


def pts_num_ctrl(pts_erase_idx, pts_add_idx):
    '''
        input : pts - to erase 
                pts - to add
        output :pts - to add (number controled)
    '''
    if len(pts_erase_idx)>=len(pts_add_idx):
        num_diff = len(pts_erase_idx)-len(pts_add_idx)
        if num_diff == 0:
            pts_add_idx_ctrled = pts_add_idx
        else:
            pts_add_idx_ctrled = np.append(pts_add_idx, pts_add_idx[np.random.randint(0, len(pts_add_idx), size=num_diff)])
    else:
        pts_add_idx_ctrled = np.sort(np.random.choice(pts_add_idx, size=len(pts_erase_idx), replace=False))
    return pts_add_idx_ctrled


def pgd(data_batch,model, task, loss_name, dataset_name, step= 7, eps=0.05, alpha=0.01):
    model.eval()
    data = data_batch['pc']
    adv_data=data.clone()
    adv_data=adv_data+(torch.rand_like(adv_data)*eps*2-eps)
    adv_data.detach()
    adv_data_batch = {}

    for _ in range(step):
        adv_data.requires_grad=True
        out = model(**{'pc':adv_data})
        adv_data_batch['pc'] = adv_data
        adv_data_batch['label'] = data_batch['label']
        model.zero_grad()
        loss = get_loss(task, loss_name, adv_data_batch, out, dataset_name)
        loss.backward()
        with torch.no_grad():
            adv_data = adv_data + alpha * adv_data.grad.sign()
            delta = adv_data-data
            delta = torch.clamp(delta,-eps,eps)
            adv_data = (data+delta).detach_()
        
        return adv_data_batch
    else:
        return data_batch


def cvar(data_batch, model, task=None, loss_name="cross_entropy", dataset_name=None, 
         step=20, 
         eps=0.05, 
         use_true=False, 
         drop_rate=100,
         expand_rate=1.5,
         min_expand_rate=0.0,
         drop_strategy="random"):
    # drop_strategy (1) random: random compelete. (2) zero: fill with 0 
    #(3) drop: just directly drop. (4) expand: multiply expand_rate
    # (5) random_expand: multiply random expand_rate

    model.eval()
    data = data_batch['pc']
    point_num = data_batch['pc'].size()[1]
    
    adv_data_=data.clone()
    ori_label = data_batch['label']
    risk = torch.zeros(adv_data_.size(0), adv_data_.size(1)).to(data)
    for _ in range(step):
        adv_data=adv_data_+(torch.rand_like(adv_data_)*eps*2-eps)
        adv_data.detach()
        adv_data.requires_grad = True
        

        out = model(**{'pc':adv_data})
        if _ == 0:
            label = ori_label if use_true else out['logit'].max(dim=-1)[1].clone().detach()
            adv_data_batch = {'pc':adv_data, 'label':label}
            continue 
        adv_data_batch['pc'] = adv_data

        model.zero_grad()
        loss = get_loss(task, loss_name, adv_data_batch, out, dataset_name)
        loss.backward()
        with torch.no_grad():
            risk += torch.norm(adv_data.grad, dim=-1)

    drop_num = drop_rate if isinstance(drop_rate, int) else int(drop_rate * point_num)
    adv_data_batch['pc'] = data
    
    if drop_strategy == "random":
        inds = torch.sort(risk, dim=1, descending=True)[1]
        drop_ind = inds[:, :drop_num]
        keep_ind = inds[:, drop_num:]
        rand_keep_ind = torch.randint(keep_ind.size(1), 
                                      size=(keep_ind.size(0), drop_num))
        rand_keep_ind = keep_ind[torch.arange(keep_ind.size(0)),
                                 rand_keep_ind.T]
        adv_data_batch['pc'].requires_grad = False
        with torch.no_grad():
          adv_data_batch['pc'][torch.arange(drop_ind.size(0)), drop_ind.T, :] = \
            adv_data_batch['pc'][torch.arange(drop_ind.size(0)), rand_keep_ind, :]

    elif drop_strategy == "zero":
        drop_ind = torch.topk(risk, k=drop_num, dim=-1)[1]
        adv_data_batch['pc'].requires_grad = False
        with torch.no_grad():
          adv_data_batch['pc'][torch.arange(drop_ind.size(0)), drop_ind.T, :] = 0.

    elif drop_strategy == "drop":
        keep_ind = torch.topk(-risk, k=point_num - drop_num, dim=-1)[1]
        adv_data_batch['pc'].requires_grad = False
        with torch.no_grad():
          adv_data_batch['pc'] = \
            adv_data_batch['pc'][torch.arange(keep_ind.size(0)), keep_ind.T, :]
        adv_data_batch['pc'] = adv_data_batch['pc'].permute(1,0,2)

    elif drop_strategy == "expand":
        drop_ind = torch.topk(risk, k=drop_num, dim=-1)[1]
        adv_data_batch['pc'].requires_grad = False
        with torch.no_grad():
          adv_data_batch['pc'][torch.arange(drop_ind.size(0)), drop_ind.T, :] = \
            adv_data_batch['pc'][torch.arange(drop_ind.size(0)), drop_ind.T, :] * expand_rate

    elif drop_strategy == "random_expand":
        drop_ind = torch.topk(risk, k=drop_num, dim=-1)[1]
        adv_data_batch['pc'].requires_grad = False
        expand_rate_ = torch.rand(drop_num, 1, 1).to(drop_ind)
        expand_rate_ = expand_rate_ * (expand_rate - min_expand_rate) + min_expand_rate
        with torch.no_grad():
          adv_data_batch['pc'][torch.arange(drop_ind.size(0)), drop_ind.T, :] = \
            adv_data_batch['pc'][torch.arange(drop_ind.size(0)), drop_ind.T, :] * expand_rate_
    adv_data_batch['label'] = ori_label 
    if RETURN_RISK: adv_data_batch.update({'risk': risk.cpu().detach().numpy()})
    return adv_data_batch


def cvar_drop_rate(epoch, max_epoch, rates=[100, 0], epoch_ratios=[ 0.5, 1.0]):
    epoch_ratio = epoch / float(max_epoch)
    assert epoch_ratio <= 1.0, 'max epoch is too large.'
    return   rates[np.digitize(epoch_ratio, epoch_ratios)]



def score_cross_entropy(logits, label):
    return F.cross_entropy(logits['logit'], label)

def score_nll(logits, label):

    x = F.log_softmax(x.view(-1,logits.shape[-1]), dim=-1)
    return F.cross_entropy(x, label)


def score_jaocbian(logits, label):
    logits = logits['logit']
    return logits[torch.arange(logits.shape[0]).to(logits).long(), label.long()].sum()


def score_sum(logits, label):
    logits = logits['logit']
    return logits.sum()

def score_borier(logits, label):

    r = logits[torch.arange(label.shape[0].to(label)), label]
    r_norm = torch.square(logits).sum(dim=-1)
    return 2*r - r_norm

def score_spherical(logits, label):

    r = logits[torch.arange(label.shape[0].to(label)), label]
    r_norm = torch.lingalg.norm(logits, dim=-1)
    return r / r_norm

def score_new(task, loss_name, adv_data_batch, out, dataset_name):
    get_loss(task, loss_name, adv_data_batch, out, dataset_name)
    

def advanced_cvar(data_batch, model, 
                  task=None, 
                  loss_name="cross_entropy", 
                  dataset_name=None, 
         step=3, 
         iter_num=10,
         eps=0.05,
         topk=1, 
         use_true=True, 
         score_function=score_cross_entropy,
         drop_rate=100,
         drop_strategy=None, **kwargs):
    point_num = data_batch['pc'].size()[1]
    all_drop_num = int(drop_rate) if drop_rate>1 else int(drop_rate * point_num)
    drop_nums = [int(all_drop_num/iter_num)+1 for _ in range(iter_num)]
    model.eval()
    ori_label = data_batch['label'].clone()
    if model.type.low() == 'pointnet':
        track_modules = ['model.feat.fstn', 'model.feat.conv3']
    elif model.type.low() == 'dgcnn':
        track_modules = ['model.conv5', 'model.conv2']
    elif model.type.low() == 'pct':
        track_modules = ['model.identity']
    elif model.type.low() == 'gdanet':
        track_modules = []
    
    for __ in range(iter_num):
        data = data_batch['pc']
        drop_num = drop_nums[__]
        adv_data_=data.clone()
        risk = 0.0
        if __ == 0: 
            out = model(**{'pc':adv_data_, 'topk':1, 'logits':True})                                                                                                                                                                
            label = ori_label if use_true else out['logit'].max(dim=-1)[1].clone().detach()
            adv_data_batch = {'label':label, 'pc':adv_data_}
            if score_function is None:
                score_function = partial(get_loss, task, loss_name, adv_data_batch, 
                                        dataset_name=dataset_name)
        risk = smooth_risk_calculation(model, adv_data_, label, eps, step, score_function, track_modules, **kwargs)    
        l_risk = calculate_local_risk(adv_data_, k=20, mode='norm').to(risk)
        risk = risk + l_risk*1.0  
        adv_data_batch['pc'] = data 
        
        if drop_strategy == "random":
            inds = torch.sort(risk, dim=1, descending=True)[1]
            drop_ind = inds[:, :drop_num]
            keep_ind = inds[:, drop_num:]
            rand_keep_ind = torch.randint(keep_ind.size(1), 
                                        size=(keep_ind.size(0), drop_num))
            rand_keep_ind = keep_ind[torch.arange(keep_ind.size(0)),
                                    rand_keep_ind.T]
            adv_data_batch['pc'].requires_grad = False
            with torch.no_grad():
                adv_data_batch['pc'][torch.arange(drop_ind.size(0)), drop_ind.T, :] = \
                    adv_data_batch['pc'][torch.arange(drop_ind.size(0)), rand_keep_ind, :]

        elif drop_strategy == "zero":
            drop_ind = torch.topk(risk, k=drop_num, dim=-1)[1]
            adv_data_batch['pc'].requires_grad = False
            with torch.no_grad():
                adv_data_batch['pc'][torch.arange(drop_ind.size(0)), drop_ind.T, :] = 0.

        elif drop_strategy == "drop":
            device = adv_data_batch['pc'].device
            keep_ind = torch.topk(-risk, k=adv_data_batch['pc'].size()[1] - drop_num, dim=-1)[1].to(device)
            adv_data_batch['pc'].requires_grad = False
            with torch.no_grad():
                adv_data_batch['pc'] = \
                    adv_data_batch['pc'][torch.arange(keep_ind.size(0)).to(device), keep_ind.T.to(device), :]
                adv_data_batch['pc'] = adv_data_batch['pc'].permute(1,0,2)
        else:
            pass 
        
        
        data_batch = adv_data_batch
    adv_data_batch['label'] = ori_label
    if RETURN_RISK: adv_data_batch.update({'risk': risk.cpu().detach().numpy()})
    return adv_data_batch



def risk_calculation(model, data, label, topk=1, score_function=None, track_modules=[]):
    track_modules = [attrgetter(module)(model) for module in track_modules]
    model.zero_grad()
    grads_in_hook = []
    def hook(module, input, output): 
        grads_in_hook.append(input[0].detach()) 
                
    handles = [module.register_full_backward_hook(hook) for module in track_modules]

    out = model(**{'pc':data, 'topk':topk, 'logits':False})    
    loss = score_function(out, label=label)
    loss.backward()
    
    risk = torch.norm(data.grad, dim=-1)
    risk /= torch.norm(risk, dim=-1, keepdim=True)
    with torch.no_grad():
        for g in grads_in_hook[::-1]:
            r = torch.norm(g, dim=1)
            r /= torch.norm(r, dim=-1, keepdim=True)
            risk += r
    
    for handle in handles:
        handle.remove()
        
    return risk 


def smooth_risk_calculation(model, data, label, eps, step=1, score_function=None, track_modules=[], **kwargs):
    data, batch_index = replicate_tensor(data, step=step)
    label = replicate_label(label, step=step).to(label)
    if step>1:
        data = data + torch.rand_like(data )*eps*2-eps 
    data = data.cuda()
    batch_index = batch_index.cuda()
    data.detach()
    data.requires_grad = True 
    if not (data.grad is None):
        data.grad.zero_()
    track_modules = [attrgetter(module)(model) for module in track_modules]
    grads_in_hook = []
    model.zero_grad()
    def hook(module, input, output): 
        grads_in_hook.append(input[0].detach())
    handles = [module.register_full_backward_hook(hook) for module in track_modules]
    
    out = model(**{'pc':data, 'logits':False}) 
    loss = score_function(out, label=label)
    loss.backward()
    risk = torch.norm(data.grad, dim=-1)
    risk /= torch.norm(risk, dim=-1, keepdim=True)
    with torch.no_grad():
        for g in grads_in_hook[::-1]:
            r = torch.norm(g, dim=1)
            r /= torch.norm(r, dim=-1, keepdim=True)
            r[torch.isnan(r)] = 0
            risk += r
    for handle in handles:
        handle.remove()
    risk = sum_same_batch(risk, batch_index=batch_index) / step

    return risk 
    

def replicate_tensor(input_tensor, step):
    new_tensor = input_tensor.repeat(step, 1, 1)
    batch_index = torch.arange(input_tensor.size(0)).repeat(step)
    return new_tensor, batch_index

def replicate_label(label, step):
    return label.repeat(step)

def sum_same_batch(input_tensor, batch_index):
    sum_tensor = torch_scatter.scatter_add(input_tensor, batch_index, dim=0)
    return sum_tensor