<br />
<p align="center">
  <img src="pointcvar-logo.png" align="center" width="40%">
  <h2 align="center"><strong>PointCVaR: Risk-optimized Outlier Removal for Robust 3D Point Cloud Classification</strong></h2>
<div align="center">
  <a href="http://xinke.li">Xinke Li<sup>1</sup></a>, Junchi Lu<sup>2</sup>, 
  <a href="https://henghuiding.github.io/">Henghui Ding<sup>2</sup></a>, 
  <a href="https://sunchangsheng.com/">Changsheng Sun<sup>1</sup></a>, 
  <a href="https://joeyzhouty.github.io/">Joey,Tianyi Zhou<sup>3,4</sup></a>, 
  <a href="https://www.nus.edu.sg/about/management/chee-yeow-meng">Yeow Meng Chee<sup>1</sup></a>
</div>

<div align="center">
  <sup>1</sup>National University of Singapore<br>
  <sup>2</sup>Nanyang Technological University<br>
  <sup>3</sup> Institute of High Performance Computing (IHPC), A*STAR<br>
  <sup>4</sup> Centre for Frontier AI Research (CFAR), A*STAR
</div>

<div align="center">
  <a href="https://arxiv.org/pdf/2307.10875.pdf"  target='_blank'>
    <img src="https://img.shields.io/badge/PDF-Download-red?logo=Adobe%20Acrobat%20Reader">
  </a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2307.10875" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/2311.12085-b31b1b?logo=arXiv&label=arXiv">
  </a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="http://xinke.li" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Project%20Page-blue?logo=Google%20Chrome&logoColor=white">
  </a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.youtube.com" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Youtube-%23ff0000?style=flat&logo=Youtube">
  </a>
</div>


## Overview
<strong>PointCVaR</strong> is a novel outlier removal method using gradient-based attribution in deep learning for <strong>robust 3D point cloud classification task</strong>. The method can effectively filter out various types of noise points in point clouds, such as [natural noise](https://arxiv.org/abs/2202.03377), [adversarial noise](https://arxiv.org/abs/1809.07016), and [backdoor noise](https://arxiv.org/abs/2103.16074). This work was published in the proceedings of <strong>AAAI 2024 (oral presentation)</strong>. This implementation provides **a general framework** for evaluating **robust methods on point cloud classification**.



## Getting Started

All implementation code has been tested on Ubuntu 18.04 with Python version 3.7.5, CUDA version 11.0 and GCC version 9.4.

### Environment

We recommend installing [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/) and creating a conda environment for running by:

```
conda create --name pointCVaR python=3.7.5
```

Activate the virtual environment and install the required libraries:

```
conda activate pointCVaR
pip install -r requirements.txt
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
```
### Data Preparation
<strong>ModelNet40</strong> and **ShapeNetPart** are utilized as the experimental datasets. Please refer to [pointcvar/README.md/Data Preparation](pointcvar/README.md#data-preparation) for the details to prepare the <strong>Training datasets</strong> and the <strong>Testing datasets with various noises</strong>.

### Model Preparation
Serveral 3D point cloud classification architectures are implemented :**PointNet and DGCNN**. Please refer to [pointcvar/README.md/Model Training](pointcvar/README.md#model-training) to train the customized model for your own.
_TODO: more models being available._

## Evaluation
Please refer to [pointcvar/README.md/Model Evaluation](pointcvar/README.md#model-evaluation) for evaluation. We also provide a script that includes integrated commands for running inference on different noisy datasets with various outlier removal methods located in `pointcvar/infer.sh`. To use this script:

1. Download all necessary resources as mentioned in the following section.
2. Run the script:
```bash
cd ./pointcvar
source infer.sh
```

## Download
| Model Name |                         Reference                         |                       Model                        |                         ModelNet40 Testing Data                          | ShapeNetPart Testing Data    |
| :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |:----------------------------------------------------------: |
|  PointNet  | [Qi et al.](https://arxiv.org/abs/1612.00593) |[Clean](https://drive.google.com/drive/folders/1ulkI1bFm7fyi2Ufktn2dK4qEzAwkXiCv?usp=sharing) / [Backdoored](https://drive.google.com/drive/folders/1If_k2e_8Im0eEbqsWZc3R310lfp7L3gX?usp=sharing) | [PointNet Data](https://drive.google.com/drive/folders/19qD6NusbgZGb_PBrLo2I1Lf60Rq5Jc5u?usp=sharing) |[PointNet Data](https://drive.google.com/drive/folders/19qD6NusbgZGb_PBrLo2I1Lf60Rq5Jc5u?usp=sharing) |
|   DGCNN    | [Wang et al.](https://arxiv.org/abs/1801.07829)  | [Clean](https://drive.google.com/drive/folders/12p5mLjALB_2VKRSN8aY4Y5zezgSE0sG0?usp=share_link) / [Backdoored](https://drive.google.com/drive/folders/1N-vjOiWOzrgvL-AA2VMQ5dyZtsxKRLRR?usp=sharing) | [DGCNN Data](https://drive.google.com/drive/folders/1-6qcvhrC0Jew-MV67bmHXfcgTHBLzTgP?usp=sharing) |[DGCNN Data](https://drive.google.com/drive/folders/1-6qcvhrC0Jew-MV67bmHXfcgTHBLzTgP?usp=sharing) |

*TODO: How to place the files.


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This work is licensed under the [MIT License](https://opensource.org/licenses/MIT).



## Acknowledgement

Our implementation codes are largely motivated by 
[ModelNet40-C](https://github.com/jiachens/ModelNet40-C) and [IF-Defense](https://github.com/Wuziyi616/IF-Defense). We also thank the authors of following works for opening source their excellent codes.

* [PointNet/PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [DGCNN](https://github.com/WangYueFt/dgcnn), [PCT](https://github.com/Strawberry-Eat-Mango/PCT_Pytorch), [GDANet](https://github.com/mutianxu/GDANet)
* [Perturb/Add attack](https://github.com/xiangchong1/3d-adv-pc), [kNN attack](https://github.com/jinyier/ai_pointnet_attack), [Drop attack](https://github.com/tianzheng4/PointCloud-Saliency-Maps), [AdvPC](https://github.com/ajhamdi/AdvPC)
* [PU-Net](https://github.com/lyqun/PU-Net_pytorch), [DUP-Net](https://github.com/RyanHangZhou/DUP-Net)

## Citation

If you find this work helpful, please kindly consider citing our papers:
```
@inproceedings{li2024pointcvar,
  title={PointCVaR: Risk-optimized Outlier Removal for Robust 3D Point Cloud Classification},
  author={Li, Xinke and Lu, Junchi and Ding, Henghui and Sun, Changsheng and Zhou, Joey Tianyi and Chee, Yeow Meng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  year={2024}
}
```



