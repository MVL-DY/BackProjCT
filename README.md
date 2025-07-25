# BackProjCT : Bi-Planar X-ray to CT Reconstruction with Back Projection Driven 3D Attention Mapping 

## Code availability
-----
We will release the code after the paper is accepted.

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
