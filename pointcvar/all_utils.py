import tensorboardX
import pdb
import sys
from collections import MutableMapping, Hashable
import csv
import os
import torch
import torch.nn.functional as F
import numpy as np
from progressbar import ProgressBar
import sys
import open3d as o3d
import h5py 
from scipy.spatial.distance import cdist

# Additional information that might be necessary to get the model
DATASET_NUM_CLASS = {
    'modelnet40_c': 40,
    'shapenet_c': 16,
    'modelnet40_dgcnn': 40,
    'shapenetpart_dgcnn': 16
}

# 上看这里记得改，对于 shapenet 是16 mn40 是 40

class TensorboardManager:
    def __init__(self, path):
        self.writer = tensorboardX.SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            self.writer.add_scalar('%s_%s' % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class TrackTrain:
    def __init__(self, early_stop_patience):
        self.early_stop_patience = early_stop_patience
        self.counter = -1
        self.best_epoch_val = -1
        self.best_epoch_train = -1
        self.best_epoch_test = -1
        self.best_val = float("-inf")
        self.best_test = float("-inf")
        self.best_train = float("-inf")
        self.test_best_val = float("-inf")

    def record_epoch(self, epoch_id, train_metric, val_metric, test_metric):
        assert epoch_id == (self.counter + 1)
        self.counter += 1

        if val_metric >= self.best_val:
            self.best_val = val_metric
            self.best_epoch_val = epoch_id
            self.test_best_val = test_metric

        if test_metric >= self.best_test:
            self.best_test = test_metric
            self.best_epoch_test = epoch_id

        if train_metric >= self.best_train:
            self.best_train = train_metric
            self.best_epoch_train = epoch_id


    def save_model(self, epoch_id, split):
        """
        Whether to save the current model or not
        :param epoch_id:
        :param split:
        :return:
        """
        assert epoch_id == self.counter
        if split == 'val':
            if self.best_epoch_val == epoch_id:
                _save_model = True
            else:
                _save_model = False
        elif split == 'test':
            if self.best_epoch_test == epoch_id:
                _save_model = True
            else:
                _save_model = False
        elif split == 'train':
            if self.best_epoch_train == epoch_id:
                _save_model = True
            else:
                _save_model = False
        else:
            assert False

        return _save_model

    def early_stop(self, epoch_id):
        assert epoch_id == self.counter
        if (epoch_id - self.best_epoch_val) > self.early_stop_patience:
            return True
        else:
            return False


class PerfTrackVal:
    """
    Records epoch wise performance for validation
    """
    def __init__(self, task, extra_param=None):
        self.task = task
        if task in ['cls', 'cls_trans']:
            assert extra_param is None
            self.all = []
            self.class_seen = None
            self.class_corr = None
        else:
            assert False
    def update(self, data_batch, out):
        if self.task in ['cls', 'cls_trans']:
            correct = self.get_correct_list(out['logit'], data_batch['label'])
            self.all.extend(correct)
            self.update_class_see_corr(out['logit'], data_batch['label'])
        else:
            assert False
    def agg(self):
        if self.task in ['cls', 'cls_trans']:
            perf = {
                'acc': self.get_avg_list(self.all),
                'class_acc': np.mean(np.array(self.class_corr) / np.array(self.class_seen,dtype=np.float))
            }
        else:
            assert False
        return perf

    def update_class_see_corr(self, logit, label):
        if self.class_seen is None:
            num_class = logit.shape[1]
            self.class_seen = [0] * num_class
            self.class_corr = [0] * num_class

        pred_label = logit.argmax(axis=1).to('cpu').tolist()
        for _pred_label, _label in zip(pred_label, label):
            self.class_seen[_label] += 1
            if _pred_label == _label:
                self.class_corr[_pred_label] += 1

    @staticmethod
    def get_correct_list(logit, label):
        label = label.to(logit.device)
        pred_class = logit.argmax(axis=1)
        return (label == pred_class).to('cpu').tolist()
    @staticmethod
    def get_avg_list(all_list):
        for x in all_list:
            assert isinstance(x, bool)
        return sum(all_list) / len(all_list)


class PerfTrackTrain(PerfTrackVal):
    """
    Records epoch wise performance during training
    """
    def __init__(self, task, extra_param=None):
        super().__init__(task, extra_param)
        # add a list to track loss
        self.all_loss = []

    def update_loss(self, loss):
        self.all_loss.append(loss.item())

    def agg_loss(self):
        # print(self.all_loss)
        return sum(self.all_loss) / len(self.all_loss)

    def update_all(self, data_batch, out, loss):
        self.update(data_batch, out)
        self.update_loss(loss)


# source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
def smooth_loss(pred, gold):
    eps = 0.2

    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()

    return loss


def rscnn_voting_evaluate_cls(loader, model, data_batch_to_points_target,
                              points_to_inp, out_to_prob, log_file):
    """
    :param loader:
    :param model:
    :param data_batch_to_points_target:
    :param points_to_inp: transform the points to input for the particular model
    that is evaluated
    :param out_to_prob:
    :return:
    """
    import rs_cnn.data.data_utils as d_utils
    import pointnet2.utils.pointnet2_utils as pointnet2_utils
    import numpy as np

    terminal = sys.stdout
    log = open(log_file, "w")

    NUM_REPEAT = 300
    NUM_VOTE = 10
    PointcloudScale = d_utils.PointcloudScale()   # initialize random scaling

    def data_aug(vote_id, pc):
        # furthest point sampling
        # (B, npoint)
        fps_idx = pointnet2_utils.furthest_point_sample(points, 1200)
        new_fps_idx = fps_idx[:, np.random.choice(1200, num_points, False)]
        new_points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), new_fps_idx).transpose(1, 2).contiguous()
        if vote_id > 0:
            pc_out = PointcloudScale(new_points)
        else:
            pc_out = pc
        return pc_out
    print(f"RSCNN EVALUATE, NUM_REPEAT {NUM_REPEAT}, NUM_VOTE {NUM_VOTE}")

    num_points = loader.dataset.num_points
    print(f"Number of points {num_points}")

    # evaluate
    sys.stdout.flush()
    PointcloudScale = d_utils.PointcloudScale()   # initialize random scaling
    model.eval()
    global_acc = 0
    with torch.no_grad():
        for i in range(NUM_REPEAT):
            preds = []
            labels = []
            for j, data in enumerate(loader, 0):
                points, target = data_batch_to_points_target(data)
                points, target = points.cuda(), target.cuda()
                pred = 0
                for v in range(NUM_VOTE):
                    new_points = data_aug(v, points)
                    inp = points_to_inp(new_points)
                    out = model(**inp)
                    prob = out_to_prob(out)
                    pred += prob
                    # pred += F.softmax(model(**inp), dim = 1)

                pred /= NUM_VOTE
                target = target.view(-1)
                _, pred_choice = torch.max(pred.data, -1)

                preds.append(pred_choice)
                labels.append(target.data)

            preds = torch.cat(preds, 0)
            labels = torch.cat(labels, 0)
            acc = (preds == labels).sum().float() / labels.numel()
            if acc > global_acc:
                global_acc = acc
            message1 = 'Repeat %3d \t Acc: %0.6f' % (i + 1, acc)
            message2 = '\nBest voting till now, acc: %0.6f' % (global_acc)
            message = f'{message1} \n {message2}'
            terminal.write(message)
            log.write(message)

    message = '\nBest voting acc: %0.6f' % (global_acc)
    terminal.write(message)
    log.write(message)

    return global_acc


# https://github.com/charlesq34/pointnet2/blob/master/evaluate.py
# https://github.com/charlesq34/pointnet2/issues/8
# we try to keep the variables names similar to the original implementation
def pn2_vote_evaluate_cls(dataloader, model, log_file, num_votes=[12]):
    from pointnet2_tf.utils import provider
    model.eval()

    terminal = sys.stdout
    log = open(log_file, "w")

    if isinstance(num_votes, list):
        pass
    else:
        num_votes = [num_votes]

    for _num_votes in num_votes:
        print(f"num_votes: {_num_votes}")

        NUM_CLASSES = DATASET_NUM_CLASS[dataloader.dataset.dataset_name]
        SHAPE_NAMES = [line.rstrip() for line in
                       open('./data/modelnet40_ply_hdf5_2048/shape_names.txt')]

        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]

        with torch.no_grad():
            for _batch_data in dataloader:
                # based on https://github.com/charlesq34/pointnet2/blob/master/evaluate.py#L125-L150
                batch_data, batch_label = np.array(_batch_data['pc'].cpu()), np.array(_batch_data['label'].cpu())
                bsize = batch_data.shape[0]
                BATCH_SIZE = batch_data.shape[0]
                NUM_POINT = batch_data.shape[1]

                batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
                for vote_idx in range(_num_votes):
                    # Shuffle point order to achieve different farthest samplings
                    shuffled_indices = np.arange(NUM_POINT)
                    np.random.shuffle(shuffled_indices)
                    rotated_data = provider.rotate_point_cloud_by_angle(
                        batch_data[:, shuffled_indices, :], vote_idx/float(_num_votes) * np.pi * 2)

                    inp = {'pc': torch.tensor(rotated_data)}
                    out =  model(**inp)
                    pred_val = np.array(out['logit'].cpu())
                    batch_pred_sum += pred_val

                pred_val = np.argmax(batch_pred_sum, 1)
                correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
                total_correct += correct
                total_seen += bsize

                for i in range(bsize):
                    l = batch_label[i]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i] == l)


            class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
            message = ""
            for i, name in enumerate(SHAPE_NAMES):
                message += f"\n {'%10s: %0.3f' % (name, class_accuracies[i])}"
            message += f"\n {'eval accuracy: %f'% (total_correct / float(total_seen))}"
            message += f"\n {'eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)))}"
            terminal.write(message)
            log.write(message)


class BackdoorAdd(object):
    def __init__(self):
        pass
    
    @staticmethod
    class TangentSphere:
        def __init__(self, pts_num=1024):
            self.sphere = BackdoorAdd.get_ball(pts_num)
            
        def __call__(self, data, rm, *args, **kwargs):
            r = kwargs.get('r', None)
            pts_num = kwargs.get('pts_num', 32)
            if r is None: 
                try:
                    r = args[0]
                except Exception as e:
                    r = 0.1
            sphere = r * self.sphere
            sphere += [-1+r, -1+r, -1+r]
            ind1 = rm.permutation(self.sphere.shape[0])[:pts_num]
            ind2 = rm.permutation(data.shape[0])[:pts_num]
            data[ind2] = sphere[ind1]
            return data 
    
    @staticmethod
    class CornerSphere:
        def __init__(self, pts_num=1024):
            self.sphere = BackdoorAdd.get_ball(pts_num)
            
        def __call__(self, data, rm, *args, **kwargs):
            r = kwargs.get('r', None)
            pts_num = kwargs.get('pts_num', 32)
            if r is None: 
                try:
                    r = args[0]
                except Exception as e:
                    r = 0.1
            sphere = r * self.sphere
            sphere += [-1, -1, -1]
            ind1 = rm.permutation(self.sphere.shape[0])[:pts_num]
            ind2 = rm.permutation(data.shape[0])[:pts_num]
            data[ind2] = sphere[ind1]
            return data 

    @staticmethod
    class OnePoint:
        def __init__(self, *args, **kwargs):
            pass 
        
        def __call__(self, data, rm, *args, **kwargs):
            loc = kwargs.get('loc', None)
            if loc is None: loc = [-1, -1, -1]
            loc = np.asarray(loc)
            ind2 = rm.permutation(data.shape[0])[0]
            data[ind2] = loc[:]
            return data 
    
    @staticmethod
    def poison_label_attack(data, label, backdoor, target_label, rate, seed=255, *args, **kwargs ):
        rm = np.random.RandomState(seed)
        data_size = data.shape[0]
        non_target_indices = np.where(label != target_label)[0]
        if rate != 1.0:
            poison_indices = rm.permutation(non_target_indices.shape[0])[:int(rate*data_size)]
            poison_indices = non_target_indices[poison_indices]
        else:
            poison_indices = np.arange(data_size)
        add_func = getattr(BackdoorAdd, backdoor)(pts_num=data.shape[1])
        for i in poison_indices:
            data[i] = add_func(data[i], rm, *args, **kwargs)
            label[i] = target_label
        return data, label
    
    @staticmethod
    def clean_label_attack(data, label, backdoor, target_label, rate, 
                            clean_target_feats,
                            corrupted_target_data,
                            corrupted_target_feats,
                           seed=255, *args, **kwargs ):
        rm = np.random.RandomState(seed)
        data_size = data.shape[0]
        target_indices = np.where(label == target_label)[0]
        if rate != 1.0:
            clean_target_data = data[target_indices]
            ind, selected_corrupted_target_data = BackdoorAdd.feature_selector(
                                         clean_target_data, 
                                         corrupted_target_data,
                                         clean_target_feats,
                                         corrupted_target_feats,
                                         ratio=rate
                                         )
            poison_indices = target_indices[ind]
            data[poison_indices] = selected_corrupted_target_data
        else:
            poison_indices = np.arange(data_size)
        add_func = getattr(BackdoorAdd, backdoor)(pts_num=data.shape[1])
        for i in poison_indices:
            data[i] = add_func(data[i], rm, *args, **kwargs)
            label[i] = target_label
        return data, label
    
    @staticmethod
    def feature_selector(data1, data2, feat1, feat2, ratio=0.5):
        #data1 is clean, data2 is distangled
        K = rbf_kernel(feat1, feat2)
        inds = np.argsort(K.sum(0))
        return inds[:int(ratio*data1.shape[0])], data2[inds[:int(ratio*data1.shape[0])]]


    @staticmethod
    def get_ball(pts_num=1024):
        m = o3d.geometry.TriangleMesh.create_sphere()
        pts = m.sample_points_uniformly(pts_num)
        return np.array(pts.points) 
    
    @staticmethod
    def get_backoored_hdf(in_h5, 
                          out_h5, 
                          backdoor='TangentSphere', 
                          attack='poison',split='train', 
                          rate=0.025, target_label=30, 
                          clean_target_feats=None,
                          corrupted_target_data=None,
                           corrupted_target_feats=None,
                          *args, **kwargs):
        with h5py.File(out_h5, "w") as f:
            data_dset = f.create_dataset("data", (0,1024, 3), maxshape=(None, 2048, 3), dtype='f')
            label_dset = f.create_dataset("label", (0, ), maxshape=(None,), dtype='i')

            cur_shape = data_dset.shape[0]
            dd = load_data(in_h5)
            new_shape = dd['data'].shape[0]
            data_dset.resize((cur_shape + new_shape), axis=0)
            if split == 'test': 
                targte_label_dset = f.create_dataset("target_label", (0, ), maxshape=(None,), dtype='i')
                rate = 1.0
                targte_label_dset.resize((cur_shape + new_shape), axis=0)
                targte_label_dset[:] = int(target_label)
            if attack == 'poison':
                data,label = BackdoorAdd.poison_label_attack(dd['data'][:, :1024, :], 
                                                dd['label'][:],
                                                backdoor,
                                                target_label,
                                                rate,
                                                *args, **kwargs 
                                                )
                if split == 'test': label = np.squeeze(dd['label'][:])
            else:
                data,label = BackdoorAdd.clean_label_attack(dd['data'][:, :1024, :], 
                                                dd['label'][:],
                                                backdoor,
                                                target_label,
                                                rate,
                                                clean_target_feats=clean_target_feats,
                                                corrupted_target_data=corrupted_target_data,
                                                corrupted_target_feats=corrupted_target_feats,
                                                *args, **kwargs 
                                                )
            data_dset[cur_shape:] = data
            label_dset.resize((cur_shape + new_shape), axis=0)
            label_dset[cur_shape:] = label

            
        print(in_h5, ' saved.')
        
def load_data(data_path, num=1024):
    if not isinstance(data_path, list): 
        data_path=[data_path]
    data_all = []
    label_all = []
    for dp in data_path:
        with h5py.File(dp, "r") as f:
            data_all.append(f['data'][:, :num, ...])
            label_all.append(f['label'][:])
            print(np.shape(data_all[-1]))

    return {'data': np.concatenate(data_all), 'label': np.squeeze(np.concatenate(label_all))}

def rbf_kernel(X, Y):

    dist = cdist(X, Y, metric='sqeuclidean')
    h = np.median(dist)
    gamma = np.sqrt(h/2)
    K = np.exp(- gamma * dist )
    return K
    
    
if __name__ == '__main__':
    import glob 
    split = 'train'
    inh5files = sorted(glob.glob("./data/modelnet40_ply_hdf5_2048/ply_data_{}*.h5".format(split)))
    '''
    out_h5 = "tangent_sphere_pl_attack_data_{}.h5".format(split)
    BackdoorAdd.get_backoored_hdf(inh5files, out_h5, backdoor='TangentSphere', split=split, rate=0.025, target_label=30, )
    out_h5 = "corner_sphere_pl_attack_data_{}.h5".format(split)
    BackdoorAdd.get_backoored_hdf(inh5files, out_h5, backdoor='CornerSphere',split=split, rate=0.025, target_label=30, )
    out_h5 = "one_point_pl_attack_data_{}.h5".format(split)
    BackdoorAdd.get_backoored_hdf(inh5files, out_h5, backdoor='OnePoint',split=split, rate=0.025, target_label=30, )
    '''
    clean_target_feats = np.load('/home/lixk/code/PointBA-1/pointnet_ori_feats_4_pointcvar.npy')
    corrupted_target_data  = np.load('/home/lixk/code/PointBA-1/pointnet_ori_points_4_pointcvar.npy')
    corrupted_target_feats  = np.load('/home/lixk/code/PointBA-1/pointnet_attc_feats_4_pointcvar.npy')

    out_h5 = "tangent_sphere_cl_attack_data_{}.h5".format(split)
    BackdoorAdd.get_backoored_hdf(inh5files, out_h5, attack='clean', backdoor='TangentSphere', split=split, rate=0.5, target_label=30, 
                                                clean_target_feats=clean_target_feats,
                                                corrupted_target_data=corrupted_target_data,
                                                corrupted_target_feats=corrupted_target_feats,)
    out_h5 = "corner_sphere_cl_attack_data_{}.h5".format(split)
    BackdoorAdd.get_backoored_hdf(inh5files, out_h5, attack='clean',backdoor='CornerSphere',split=split, rate=0.5, target_label=30, 
                                                 clean_target_feats=clean_target_feats,
                                                corrupted_target_data=corrupted_target_data,
                                                corrupted_target_feats=corrupted_target_feats,)
    out_h5 = "one_point_cl_attack_data_{}.h5".format(split)
    BackdoorAdd.get_backoored_hdf(inh5files, out_h5, attack='clean',backdoor='OnePoint',split=split, rate=0.5, target_label=30, 
                                                clean_target_feats=clean_target_feats,
                                                corrupted_target_data=corrupted_target_data,
                                                corrupted_target_feats=corrupted_target_feats,)
    
