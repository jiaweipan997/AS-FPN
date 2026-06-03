# AS-FPN
This repo is the official implementation of "AS-FPN: An Asymmetric Semantic-Preserving Feature Pyramid Network for Efficient Semantic Segmentation". This repo contains the supported code, configuration files, and datasets to reproduce the semantic segmentation results of AS-FPN. The code is mainly based on [MMSegmentaion V1.2.2.](https://github.com/open-mmlab/mmsegmentation/tree/main) All experiments were performed on NVIDIA GTX 3090Ti GPUs in CUDA 12.1, Python 3.11, and PyTorch 2.4.1.

## Code Snippet
The code snippet is [here](mmseg/models/necks/asfpn.py).

## Citation
If you find our repo useful for your research, please consider citing our paper.

## Installation
Step 1. Create a conda environment and activate it.
```
conda create --name ASFPN python=3.11 -y
conda activate ASFPN
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
cd ASFPN
pip install -v -e .
pip install ftfy
pip install regex
pip install mmpretrain>=1.0.0rc7
```

## Dataset prepare
Dataset preparation：https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets

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
