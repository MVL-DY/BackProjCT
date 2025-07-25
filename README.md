# BackProjCT : Bi-Planar X-ray to CT Reconstruction with Back Projection Driven 3D Attention Mapping 

## Introduction
-----
This is the official code release of the 2026 [TBD] paper BackProjCT : Bi-Planar X-ray to CT Reconstruction with Back Projection Driven 3D Attention Mapping. In the original paper, we present a novel framework for reconstructing volumetric CT images from bi-planar X-rays, addressing the challenge of spatial ambiguity inherent in limited 2D projections. Our approach integrates differentiable back projection for coarse volumetric initialization, attention-guided 2D-to-3D feature mapping, and transformer-based 3D bottleneck refinement. This design enables anatomically accurate, perceptually realistic CT reconstructions while maintaining computational efficiency. On the LIDC dataset, our model achieves state-of-the-art performance, including a PSNR of 26.95 dB and the lowest LPIPS of 0.1080, surpassing GAN- and diffusion-based baselines. The proposed method demonstrates strong potential for low-dose, real-time CT imaging applications in clinical practice.

![Image](https://github.com/user-attachments/assets/921372f9-d784-4f3f-b0a6-346bfee9c790)

### License
This work is released under the GPLv3 license (refer to the LICENSE file for more details).

## Contents
----
1. [Requirements](#Requirements)
2. [Installation](#Installation)
3. [Code Structure](#Structure)
4. [Results](#Results)
5. [Acknowledgement](#Acknowledgement)
6. [Citations](#Citations)

## Requirements
----
1. pytorch>=2.4.1 versions had been tested 
3. python3.10.18 was tested
4. python dependencies, please see the requirements.txt file
5. CUDA12.1 and cudnn 12.8 had been tested

## Installation
----
- Install conda enviroment
```
   conda env create -f environment.yaml
   conda activate BPJCT
```
- Make sure PyTorch >= 2.4.1, CUDA >= 12.1 and cudnn are installed
- Install DRR Projection via back projector 
```
   cd drr_projector
   python setup.py install
   cd ../
```
- Download preprocessed X2CT-GAN dataset and our model files: <a href="https://share.weiyun.com/5xRVfvP">weiyun</a> or <a href="https://www.dropbox.com/sh/06r3g02dyeyh5x5/AADFhwRuG_SPuGGwKx-SZLrna?dl=0">dropbox</a>
- Download the source code and put the data file to the right location according to the code structure below

## Structure
----
```
BackProjCT/:
   |--data/:folder include all the preprocessed data and train/test split in our original experiment
   |    |--LIDC-HDF5-256/:include the raw data .h5 file
   |    |--train.txt:training file list
   |    |--test.txt:test file list
   |
   |--drr_projector/: Source code for back projector
   |    |--dist/: building code for porjector
   |
   |--experiment/: experiment configuration folder
   |    |--multiView2500_BP/: multiview experiment configuration file
   |
   |--lib/:folder include all the dependency source codes
   |    |--config/: folder includes the config file
   |    |--dataset/: folder includes the source code to process the data
   |    |--model/: folder includes the network definitions and loss definitions
   |    |--utils/: utility functions
   |
   |--save_models/: folders include trained modesl from us
   |    |--multiView_CTGAN/: multi view X-Ray to CT model
   |  
   |--test.py: test script that demonstrates the inference workflow and outputs the metric results
   |--train.py: training script that trains models
   |--visual.py: same working mechanism as test.py but viualizing the output instead of calculating the statistics 
   |--README.md
```

### Input Arguments
+ --ymlpath: path to the configuration file of the experiemnt
+ --gpu: specific which gpu device is used for testing, multiple devices use "," to separate, e.g. --gpu 0,1,2,3
+ --dataroot: path to the test data
+ --dataset: flag indicating data is for training, validation or testing purpose
+ --tag: name of the experiment that includes the trained model
+ --data: input dataset prefix for saving and loading purposes, e.g. LIDC256 
+ --dataset_class: input data format, e.g. single view X-Rays or multiview X-Rays, see lib/dataset/factory.py for the complete list of supported data input format
+ --model_class: flag indicating the selected model, see lib/model/factory.py for the complete list of supported models
+ --datasetfile: the file list used for testing
+ --resultdir: output path of the algorithm
+ --check_point: the selected training iteration to load the correct checkpoint of the model
+ --how_many: how many test samples will be run for visualization (useful for visual mode only)
+ --valid_datasetfile: the file list used for validation

### Test our Models

Please use the following example settings to test our model. 
 
- **Test Script：**  
python3 test.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0 --dataroot=../../Datasets/LIDC-HDF5-256/LIDC-HDF5-256 --dataset=test --tag=d2_multiview2500 --data=LIDC256 --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=../../Datasets/LIDC-HDF5-256/test.txt --resultdir=./multiview --check_point=100 --load_path=./save_models/multiView_CTGAN/LIDC256/BPJCT_LIDC/checkpoint

### Train from Scratch
Please use the following example settings to train your model. 

- **Training Script：**  
python3 train.py --ymlpath=./experiment/multiview2500/d2_multiview2500_BP.yml --gpu=0 --dataroot=../../Datasets/LIDC-HDF5-256/LIDC-HDF5-256 --dataset=train --tag=d2_multiview2500 --data=LIDC256 --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=../../Datasets/LIDC-HDF5-256/train.txt --valid_datasetfile=../../Datasets/LIDC-HDF5-256/test.txt --valid_dataset=test

## Results
----
Qualitative results from our original paper. <br>

![Image](https://github.com/user-attachments/assets/1aeb1709-aa96-4f40-9ea6-75a92d3ed02f)
![Image](https://github.com/user-attachments/assets/20571a6d-e0b8-4150-8b18-fe88ebf99466)

## Acknowledgement
----
We thank the public <a href="https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI">LIDC-IDRI dataset </a> that is used to build our algorithm. 

## Citations
```
@software{Ying_2019_CVPR,
author = {Ying, Xingde and Guo, Heng and Ma, Kai and Wu, Jian and Weng, Zhengxin and Zheng, Yefeng},
title = {X2CT-GAN: Reconstructing CT From Biplanar X-Rays With Generative Adversarial Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}}
```
