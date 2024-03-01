import os
import numpy as np
import argparse
import h5py
LABEL_FILE = './data/modelnet40_c/label.npy'


def load_npz(file_name, pc_key="test_pc", label_key="test_label", target_key="target_label"):
    assert os.path.isfile(file_name), "File loading failed."
    n = np.load(file_name)
    if target_key in n.keys():
        return {'pc':n[pc_key], 'label':n[label_key], 'target':n[target_key]}
    else:
        return {'pc':n[pc_key], 'label':n[label_key]}

def load_h5(file_name, pc_key="data", label_key="label", use_modelnet40clabel=False): #, target_key="target_label"
    #assert os.path.isfile(file_name), "File loading failed."
    file_name = file_name.split(' ')
    if len(file_name) == 0: file_name = file_name[0]
    if isinstance(file_name, list):
        n = load_multi_h5(file_name)
    else:
        n = h5py.File(file_name, 'r')
    if use_modelnet40clabel:
        d= {'pc':n[pc_key][:], 'label':np.load(LABEL_FILE), 'target':n[target_key][:]}
    else:
        d= {'pc':n[pc_key][:], 'label':n[label_key][:]}
    if not isinstance(file_name, list): n.close()
    return d

def load_multi_h5(file_names):
    d = None 
    for f in file_names:
        with h5py.File(f, 'r') as hf:
            if d is None:
                d  = {k:[] for k in hf.keys()}
            for k in d.keys():
                d[k].append(hf[k][:])
    d = {k:np.concatenate(v) for k, v in d.items()}
    return d 


def save_npy(dir_name, save_dict, corruption, severity):
    DATA_DIR = os.path.join(dir_name, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(dir_name, 'label_'+ corruption + '_' +str(severity) + '.npy')
    TARGET_LABEL_DIR = os.path.join(dir_name, 'target_label_'+ corruption + '_' +str(severity) + '.npy')
    np.save(DATA_DIR, save_dict['pc'])
    print(DATA_DIR, ' saved.')
    np.save(LABEL_DIR, save_dict['label'])
    print(LABEL_DIR, ' saved.')
    if 'target' in save_dict.keys():
        np.save(TARGET_LABEL_DIR, save_dict['target'])
        print(TARGET_LABEL_DIR, ' saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument('--file', type=str, default="")
    parser.add_argument('--dir', type=str, default="./data/modelnet40_c")
    parser.add_argument('--corruption', type=str, default="")
    parser.add_argument('--severity', type=int, default=1)
    parser.add_argument('--use_modelnet40clabel', '-u', type=bool, default=False)
    
    cmd_args = parser.parse_args()
    if os.path.splitext(cmd_args.file)[-1] == '.h5':
        load_dict = load_h5(cmd_args.file) #, cmd_args.use_modelnet40clabel
    else:
        load_dict = load_npz(cmd_args.file)
    save_npy(cmd_args.dir, load_dict, cmd_args.corruption, cmd_args.severity)
    
    
    

