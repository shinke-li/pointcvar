# Code for Robust Point Cloud Classification

## Data Preparation
All dataset files should be placed in the `./data` directory. To create this directory, navigate to the current working directory, which is `<ROOT>/pointcvar/pointcvar`, and run the following command:
```bash
mkdir ./data
```



### Training Dataset
We utilize the data processed by [Qi et al.](https://arxiv.org/abs/1612.00593) for ModelNet40. Using the similar method, ShapeNetPart is also processed. Two datasets are publicly available from the links: 
|ModelNet40 | ShapeNetPart |
|:---:|:---:|
| [link](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)| [link](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip) |

The downloading should be conducted in `./data` directory as:
```bash
cd ./data
wget <weblink to dataset>
```
To extract the downloaded file for ModelNet40:
```bash
unzip modelnet40_ply_hdf5_2048.zip
rm modelnet40_ply_hdf5_2048.zip
```
To extract the downloaded file for ShapeNetPart:
```bash
unzip shapenet_part_seg_hdf5_data.zip
rm shapenet_part_seg_hdf5_data.zip
```
*TODO: add backdoor dataset*
<!-- #### Backdoor Data
We follow the [reimplementation code]() of **PointBA** and provide the links for downloading training datasets:
|ModelNet40 | ShapeNetPart |
|:---:|:---:|
| [link](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)| [link](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip) | -->


### Testing Dataset
Test data should be placed in `.npy` format inside the `./data` directory.

#### Data with noise
- **Natural noise** we employ and adapt the datasets with **add local** and **add global** noise from [PointCloud-C dataset](https://pointcloud-c.github.io/download.html). 

- **Adversarial noise** Please refer to the [attack/README.md](../attack/README.md) for specific generation steps of adversarial data.


- **Backdoor noise** We follow the [implementation code]() of PointBA to generate the test dataset.

Above testing datasets are publicly provided in [README.md/Download](../README.md#download). The downloading should be conducted in `./data` directory as:
```bash
cd ./data
wget <weblink to test data.zip>
unzip <test data.zip>
rm <test data.zip>
```

#### Customized testing data preparation

We also provide code for customized testing data preparation by converting the ``.h5`` or ``.npz`` file to the ``.npy`` file. 

```bash
python trans_to_testdata.py --file <original_data> --corruption <noise_type> --severity <noise_level>
```

- `<original_data>`: Specify the file for conversion to `.npy`.
- `<tnoise_type>`: Indicate the noise name for testing.
- `<noise_level>`: Set the noise level (higher values correspond to more noise). If omitted, the default severity is 5.
For instance, to convert a file to `.npy` format for testing with a specified type and severity of corruption, use the following command:

```bash 
python trans_to_testdata.py --file mydata.npz --corruption myadv --severity 5
```


This will produce `data_myadv_5.npy` and `label_myadv_5.npy` in the `./data` directory. Please ensure that the input file contains the keys (dataset names in `.h5`) "data" and "label", which are used to store the point cloud data and label information, respectively.


## Model Training

#### Vanilla Training
To train your own model, follow the instructions below. All training should be done within the `<ROOT>/pointcvar/pointcvar/` directory. The `./main.py` script is utilized with the following format:

```bash
python main.py --exp-config <path to config file>
```
Ready-for-run configuration files are stored in `./configs` with the naming pattern ``<model_name>_<dataset_name>.yaml``, where ``<model_name>`` can be one of the following: **pointnet**, **dgcnn**.  For instance, run the command to train PointNet on ModelNet40 
```bash
python main.py --exp-config configs/pointnet_modelnet40.yaml
```
Specifically, `configs/pointnet_modelnet40.yaml` is with the following format
```yaml
DATALOADER:
  batch_size: 32
  num_workers: 0
EXP:
  DATASET: modelnet40_dgcnn 
  EXP_ID: pointnet_modelnet40
  MODEL_NAME: pointnet
  TASK: cls_trans
TRAIN:
 l2: 1e-4
```
- **EXP_ID**: name of the experiment directory
- **DATASET**: dataset name, set *modelnet40_dgcnn* for **ModelNet40**, and *shapenetpart_dgcnn* for **ShapeNetPart**.
- **MODEL_NAME**: model name can be *pointnet* or *dgcnn*.

The trained models will be saved in `./runs/{EXP_ID}/`.
*TODO: add more models*



#### Training with robust methods

Model can be trained by robust training method **PointCutMix** and **PGD adversarial training**, of which the config files are stored in `configs/cutmix` and `configs/pgd`, respectively. To train dgcnn with PointCutMix-R on ModelNet40, please use the following command:
```bash
python main.py --exp-config configs/cutmix/dgcnn_modelnet40_r.yaml
```
*TODO: rename the training config file*


## Model Evaluation
#### Clean data model testing
To test a trained model on clean dataset, please run
```bash 
python main.py --entry test --model-path <model_file> --exp-config <path to config file>
```
 ``<model_file>`` is the trained model file. For example
```bash
python main.py --entry test --model-path ./runs/pointnet_modelnet40.pth --exp-config configs/pointnet_modelnet40.yaml
```
*TODO: change pretrained model path.*
#### Testing model on data with noise 
To test a trained model on the noisy data, using
```bash
python main.py --entry test --model-path <model_file>  --exp-config configs/corruption/<model_name>.yaml --severity <noise_level> --corruption <noise_type> --output <output_file>
```
- ``<model_name>``: could be *pointnet*, *dgcnn*.
- `<noise_level>`: the noise level, if not indicated, the default value is 5.
- `<noise_type>`: the type of noise.
- `<output_file>`: text file to store the evaluation result. 

The `<noise_level>` and `<noise_type>` are actually indicating a `<noise_type>_<noise_level>.npy` data in `./data` directory. For example, to evaluate a trained PointNet on the data with cluster noise (data file is `./data/add_cluster_5.npy`), run the following
```bash
python main.py --entry test --model-path ./runs/pointnet_modelnet40.pth   --exp-config configs/corruption/pointnet.yaml --severity 5 --corruption add_cluster --output pointnet_modelnet40_add_cluster.txt
```
*TODO: description of the downloaded test files*

#### Testing model with outlier removal
Outlier removal method can be simply implemented by indicate the config file of `--extra-config` argument in the run command based on the above testing commands.  Supported methods with their config files are
|Method|Config File|
|:--:|:--:|
|SRS|*configs/infer/srs.yaml*| 
|ROR|*configs/infer/srs.yaml*| 
|SOR|*configs/infer/srs.yaml*| 
|Vanilla PointCVaR|*configs/infer/cvar.yaml*| 
|Multistep PointCVaR|*configs/infer/iter_cvar.yaml*| 
|DUP-Net+PointCVaR|*configs/infer/dup_cvar.yaml*| 

**Example.1** Run to evaluate ROR method on clean dataset for PointNet
```bash
python main.py --entry test --model-path ./runs/pointnet_modelnet40.pth --exp-config configs/pointnet_modelnet40.yaml --output pointnet_modelnet40_ror.txt --extra-config configs/infer/ror.yaml
```
**Example.2** Run to evaluate Vanilla PointCVaR method on **dataset with adversarial points by Chamfer Distance** for PointNet
```bash
python main.py --entry test --model-path ./runs/pointnet_modelnet40.pth  --exp-config configs/corruption/pointnet.yaml --severity 5 --corruption add_chamfer --output pointnet_modelnet40_add_cluster_cvar.txt --extra-config configs/infer/cvar.yaml
```

**Example.3** Run to evaluate Multistep PointCVaR method on **dataset with adversarially added clusters** for DGCNN
```bash
python main.py --entry test --model-path ./runs/pointnet_modelnet40.pth  --exp-config configs/corruption/pointnet.yaml --severity 5 --corruption add_cluster --output pointnet_modelnet40_add_cluster_cvar.txt --extra-config configs/infer/cvar.yaml
```

More examples are given in `infer.sh`.