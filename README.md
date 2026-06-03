# AS-FPN
This repo is the official implementation of "AS-FPN: An Asymmetric Semantic-Preserving Feature Pyramid Network for Efficient Semantic Segmentation". This repo contains the supported code, configuration files, and datasets to reproduce the semantic segmentation results of AS-FPN. The code is mainly based on [MMSegmentaion V1.2.2.](https://github.com/open-mmlab/mmsegmentation/tree/main) All experiments were performed on NVIDIA GTX 3090Ti GPUs in CUDA 12.1, Python 3.11, and PyTorch 2.4.1.

## Code Snippet
The code snippet is [here](mmseg/models/necks/asfpn.py).

## Citation
If you find our repo useful for your research, please consider citing our paper.

## Installation
Step 1. Create a conda environment and activate it.
```
conda create --name AS-FPN python=3.11 -y
conda activate AS-FPN
```
Step 2. Install PyTorch following [official instructions](https://pytorch.org/get-started/previous-versions/), e.g.
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
Step 3. Install MMCV using MIM.
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
Step 4. Install BiDNet.
```
git clone -b main https://github.com/jiaweipan997/AS-FPN.git
cd AS-FPN
pip install -v -e .
pip install ftfy
pip install regex
pip install mmpretrain>=1.0.0rc7
```

## Dataset prepare
Dataset preparation：https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets
e.g.
Please create a new data folder and put the downloaded dataset in it and unzip it. The structure is as follows：
```
AS-FPN
├── mmseg
├── tools
├── configs
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
     .
     .
     .
```
### Cityscapes
The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.

By convention, **labelTrainIds.png are used for cityscapes training. We provided a script based on cityscapesscriptsto generate **labelTrainIds.png.
```
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8
```
### ADE20K
The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip). We may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

### COCO Stuff 164k
For COCO Stuff 164k dataset, please run the following commands to download and convert the augmented dataset.
```
# download
mkdir coco_stuff164k && cd coco_stuff164k
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# unzip
unzip train2017.zip -d images/
unzip val2017.zip -d images/
unzip stuffthingmaps_trainval2017.zip -d annotations/

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/coco_stuff164k.py /path/to/coco_stuff164k --nproc 8
```
By convention, mask labels in /path/to/coco_stuff164k/annotations/*2017/*_labelTrainIds.png are used for COCO Stuff 164k training and testing.

The details of this dataset could be found at [here](https://github.com/nightrome/cocostuff#downloads).

## Training
```
# AED20K dataset:
bash tools/dist_train.sh configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512-asfpn.py 8
bash tools/dist_train.sh configs/segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512-asfpn.py 1
bash tools/dist_train.sh configs/convnext/convnext-tiny_upernet_8xb2-amp-160k_ade20k-512x512-asfpn.py 8

# COCO-stuff-164K dataset:
bash tools/dist_train.sh configs/segformer/segformer_mit-b0_8xb2-80k_coco164k-512x512-asfpn.py 8
bash tools/dist_train.sh configs/segnext/segnext_mscan-t_1xb16-adamw-80k_coco164k-512x512-asfpn.py 1
bash tools/dist_train.sh configs/convnext/convnext-tiny_upernet_8xb2-amp-80k_coco164k-512x512-asfpn.py 8

# Cityscapes dataset:
bash tools/dist_train.sh configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024-asfpn.py 8
bash tools/dist_train.sh configs/segnext/segnext_mscan-t_1xb8-adamw-160k_cityscapes-512x512-asfpn.py 1
bash tools/dist_train.sh /data/pjw/ASFPN/configs/convnext/convnext-tiny_upernet_4xb2-amp-160k_cityscapes-512x1024-asfpn.py 4
```

## Evaluation
```
python tools/test.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512-asfpn.py work_dirs/segformer_mit-b0_8xb2-160k_ade20k-512x512-asfpn/iter_160000.pth
.
.
.
```

## Params and FLOPs
```
python tools/analysis_tools/get_flops.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512-asfpn.py --shape 512
python tools/analysis_tools/get_flops.py configs/segformer/segformer_mit-b0_8xb2-80k_coco164k-512x512-asfpn.py  --shape 512
python tools/analysis_tools/get_flops.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024-asfpn.py  --shape 2048 1024
.
.
.
```
