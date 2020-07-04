<img src='paper/shinjuku.jpg' align="left" width=1000>

<hr/>

# Building Upon White-Box Cartoonization as Described in CVRP2020 Paper

## Original Paper and Source

**Learning to Cartoonize Using White-box Cartoon Representations**  
*Output examples can be found at both of the following links:*
[original project page](https://systemerrorwang.github.io/White-box-Cartoonization/) | [paper](https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf)

## Usage

### Installation

**Prerequisites (Windows):**
- NVIDIA CUDA and CuDNN
- MSVC 2015 (found within Microsoft C++ Build Tools)
- Python 3.6 is the latest version compatible with the required tensorflow versions

```powershell
> python36 -m venv .\wbcvenv
> .\wbcvenv\Scripts\activate
> python36 -m pip install --upgrade pip
> python36 -m pip install --upgrade tensorflow==1.12.0
> python36 -m pip install --upgrade tensorflow-gpu==1.12.0
> python36 -m pip install --upgrade scikit-image==0.14.5
> python36 -m pip install --upgrade opencv-python
> python36 -m pip install --upgrade tqdm
```

### Inference with Pre-trained Model

- Store test images in /test_code/test_images
- Run ./cartoonize.py
- Results will be saved in /test_code/cartoonized_images

### Train

- Place your training data in corresponding folders in /dataset
- Run pretrain.py, results will be saved in /pretrain folder
- Run train.py, results will be saved in /train_cartoon folder
- Codes are cleaned from production environment and untested
- There may be minor problems but should be easy to resolve
- Pretrained VGG_19 model can be found at following [here](https://drive.google.com/file/d/1j0jDENjdwxCDb36meP6-u5xDBzmKBOjJ/view?usp=sharing) (link provided by SystemErrorWang).

### Datasets

- Due to copyright issues, we cannot provide cartoon images used for training
- However, these training datasets are easy to prepare
- Scenery images are collected from Shinkai Makoto, Miyazaki Hayao and Hosoda Mamoru films
- Clip films into frames and random crop and resize to 256x256
- Portrait images are from Kyoto animations and PA Works
- We use this [repo](https://github.com/nagadomi/lbpcascade_animeface) to detect facial areas
- Manual data cleaning will greatly increace both datasets quality

<!-- ## License

TODO replace as necessary

[]: # Copyright (C) Xinrui Wang, Jinze Yu. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

-->

<!-- ## Citation

TODO replace as necessary

If you use this code for your research, please cite our [paper](https://systemerrorwang.github.io/White-box-Cartoonization/):

@InProceedings{Wang_2020_CVPR,
author = {Wang, Xinrui and Yu, Jinze},
title = {Learning to Cartoonize Using White-Box Cartoon Representations},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

-->
