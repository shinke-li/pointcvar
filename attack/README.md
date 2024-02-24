# Adversarial noise generation

This folder contains some basic files for studying 3D adversarial attack in point cloud. This codebase is highly extendable. You can easily add more attacks based on it.

## Get started

***

### Data Storage

In order to simplify the process of data loading, I convert all the data (e.g. point cloud, GT label, target label) into a NumPy npz file. For example, for the ModelNet40 npz file `data/MN40_random_2048.npz`, it contains the following data:

- **train/test_pc**: point clouds for training and testing, shape [num_data, num_points, 3]
- **train/test_label**: ground-truth labels for training and testing data
- **target_label**: pre-assigned target labels for targeted attack

### Configurations

Some commonly-used settings are hard-coded in `config.py`. For example, the pre-trained weights of the victim models, the batch size used in model evaluation and attacks. **Please change this file if you want to use your own settings.**

## requirements

***

### Data Preparation

Please download the dataset used for attacking [here](), Uncompress it to `data/`.

Data used for attack contains only test data and each point cloud has 1024 points, the pre-assigned target label is also in it.

* `modelnet40.npz` is the attack data used for the model trained on **Modelnet40**.
* `shapenet.npz` is the attack data used for the model trained on **Shapenet**.

### Pre-trained Victim Models

We provided the pre-trained weights for the victim models used in our experiments. Download from [here]() and uncompress them into `pretrain/`. 

You can also train your own victim models according to the instructions in Victim model preparation in the README.md file in the parent folder. Then, place the trained model in the `pretrain/` directory. For example, if you've trained your model on ModelNet40, place it in `pretrain/mn40/`.

**Note that**, if you want to use your own model/weight, please modify the variable called 'BEST_WEIGHTS' in `config.py`.

## Attack

***

We implement **Perturb**, **Add Point**, **Add Cluster**, **Add Object**, **kNN**, **Advpc** and **Drop** attack. The attack scripts are in `attack_scripts/` folder. For detailed usage of each attack method, please refer to `command.txt`.
