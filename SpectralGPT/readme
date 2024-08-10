<div align="center">
<h1>SpectralGPT: Spectral Remote Sensing Foundation Model</h1>
  
This is the official repository for the paper 
"_SpectralGPT: Spectral Remote Sensing Foundation Model_".  

**IEEE TPAMI: (https://ieeexplore.ieee.org/document/10490262)

[Danfeng Hong](https://scholar.google.com/citations?hl=en&user=n7gL0_IAAAAJ&view_op=list_works&sortby=pubdate), [Bing Zhang](https://scholar.google.com/citations?user=nHup8tQAAAAJ&hl=en), Xuyang Li, Yuxuan Li, Chenyu Li, [Jing Yao](https://scholar.google.com/citations?hl=en&user=1SHd5ygAAAAJ), [Naoto Yokoya](https://scholar.google.com/citations?user=DJ2KOn8AAAAJ&hl=en), [Hao Li](https://scholar.google.com/citations?hl=en&user=yECGOCwAAAAJ), [Pedram Ghamisi](https://scholar.google.com/citations?user=Gr9afd0AAAAJ&hl=en), [Xiuping Jia](https://scholar.google.com/citations?user=-vl0ZSEAAAAJ&hl=en), [Antonio Plaza](https://scholar.google.com/citations?user=F1UAj8oAAAAJ&hl=en), [Paolo Gamba](https://scholar.google.com/citations?user=9fiHXGwAAAAJ&hl=en), [Jon Atli Benediktsson](https://scholar.google.com/citations?user=C6d7qe0AAAAJ&hl=en), [Jocelyn Chanussot](https://scholar.google.com/citations?user=6owK2OQAAAAJ&hl=en)
</div>

## Abstract
The foundation model has recently garnered significant attention due to its potential to revolutionize the field of visual representation learning in a self-supervised manner. While most foundation models are tailored to effectively process RGB images for various visual tasks, there is a noticeable gap in research focused on spectral data, which offers valuable information for scene understanding, especially in remote sensing (RS) applications. To fill this gap, we created for the first time a universal RS foundation model, named SpectralGPT, which is purpose-built to handle spectral RS images using a novel 3D generative pretrained transformer (GPT). Compared to existing foundation models, SpectralGPT 1) accommodates input images with varying sizes, resolutions, time series, and regions in a progressive training fashion, enabling full utilization of extensive RS big data; 2) leverages 3D token generation for spatial-spectral coupling; 3) captures spectrally sequential patterns via multi-target reconstruction; 4) trains on one million spectral RS images, yielding models with over 600 million parameters. Our evaluation highlights significant performance improvements with pretrained SpectralGPT models, signifying substantial potential in advancing spectral RS big data applications within the field of geoscience across four downstream tasks:  single/multi-label scene classification, semantic segmentation, and change detection.

## Overview
![alt text](./workflow-spectralGPT.jpg)

## Preparation
Install Python dependencies by running:
```shell
pip install -r requirements.txt
```
## Training SpectralGPT
The pretraining experiments were run on 8 NVIDIA GeForce RTX 4090 GPUs.

### Pretrain Dataset: fMoW-Sentinel & BigEarthNet 
You can download the official fMoW-Sentinel dataset [here](https://purl.stanford.edu/vg497cb6002). 
Try this [link](https://searchworks.stanford.edu/view/vg497cb6002) if the previous one doesn't display correctly.

You can download the official BigEarthNet dataset [here](https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz). 

### Finetune Dataset: EuroSAT & OSCD & SegMunich
You can download the official EuroSAT dataset [here](https://github.com/phelber/EuroSAT#eurosat-land-use-and-land-cover-classification-with-sentinel-2) for finetuning the pretrained model on classification tasks.  

You can download the official OSCD dataset [here](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) for finetuning the pretrained model on change detection tasks. 

You can download the official SegMunich dataset we collected [here](https://pan.baidu.com/s/1ouz_FVOdENjkAZRajjkjVw?pwd=994z) or [here](https://huggingface.co/datasets/Moonboy12138/SegMunich/blob/main/TUM_128.zip) for finetuning the pretrained model on semantic segmentation tasks.

Dataset                  |Use| Link |  
---------------------- | -------------- | -------- 
fMoW-Sentinel  |     pretrain     | [download](https://purl.stanford.edu/vg497cb6002) 
BigEarthNet |    pretrain & finetune    | [download](https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz) |    
EuroSAT  |     finetune     | [download](https://github.com/phelber/EuroSAT#eurosat-land-use-and-land-cover-classification-with-sentinel-2) 
OSCD  |       finetune   | [download](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) 
SegMunich  |         finetune | [download](https://pan.baidu.com/s/1ouz_FVOdENjkAZRajjkjVw?pwd=994z)  

### Pretraining
For pretraining on fMoW-Sentinel Dataset, this is the default command:
```shell
torchrun --nproc_per_node=8 main_pretrain.py \
--master_port=29501 \
--wandb spectralgpt_pretrain_stage1 \
--batch_size 16 --accum_iter 32 --blr 0.0002 \
--epochs 200 --warmup_epochs 20 --num_workers 16 \
--input_size 96 --patch_size 8 \
--mask_ratio 0.90 \
--model_type tensor \
--model mae_vit_base_patch8_96 \
--dataset_type sentinel --dropped_bands 10 \
--train_path .txt_file/train_result_demo.csv \
--output_dir .experiments/pretrain_fmow \
--log_dir .experiments/pretrain_fmow
```

For continual pretraining on BigEarthNet Dataset, this is the default command:
```shell
torchrun --nproc_per_node=8 main_pretrain.py \
--master_port=29502 \
--wandb spectralgpt_pretrain_stage2 \
--batch_size 16 --accum_iter 32 --blr 0.0001 \
--epochs 200 --warmup_epochs 20 --num_workers 16 \
--input_size 128 --patch_size 8 \
--mask_ratio 0.90 \
--model_type tensor \
--dataset_type bigearthnet \
--model mae_vit_base_patch8_128 \
--train_path .txt_file/bigearthnet_pretrain_result_demo.csv \
--resume_different_size .experiments/pretrain_fmow/checkpoint-199.pth \
--output_dir .experiments/pretrain_BEN \
--log_dir .experiments/pretrain_BEN
```

To resume a pretraining job, you can use `--resume PATH/TO/CKPT.PTH` 
(eg: `--resume .experiments/pretrain/checkpoint-10.pth`).


### Finetuning
To finetune on EuroSAT, the basic command is:
```shell
torchrun --nproc_per_node=2 main_finetune.py \
--wandb eurosat_finetune \
--batch_size 16 --accum_iter 8 --blr 0.0002 \
--epochs 150 --num_workers 16 \
--input_size 128 --patch_size 8  \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
--model_type tensor \
--model mae_vit_base_patch8_128 \
--dataset_type euro_sat --dropped_bands 10 \
--train_path .txt_file/train_euro_result.txt \
--test_path .txt_file/val_euro_result.txt \
--output_dir /home/experiments/finetune/eurosat \
--log_dir ./experiments/finetune/eurosat \
--finetune ./experiments/pretain/SpectralGPT+.pth
```


To finetune on BigEarthNet, please replace `engine_finetune`(line 44-45) with `engine_finetune_BE`(line 46-47) in the [main_finetune.py](./main_finetune.py), the basic command is:
```shell
torchrun --nproc_per_node=2 main_finetune.py \
--wandb bigearthnet_finetune \
--batch_size 16 --accum_iter 8 --blr 0.0002 \
--epochs 150 --num_workers 16 \
--input_size 128 --patch_size 8  \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
--model_type tensor \
--model mae_vit_base_patch8_128 \
--dataset_type euro_sat --dropped_bands 10 \
--train_path .txt_file/bigearthnet_train.txt \
--test_path .txt_file/bigearthnet_val.txt \
--output_dir /home/experiments/finetune/BEN \
--log_dir ./experiments/finetune/BEN \
--finetune ./experiments/pretain/SpectralGPT+.pth
```

We also released the codes of change detection on OSCD and semantic segmentation on SegMunich in the [downstream_tasks](./downstream_tasks/) folder.  These codes are easy to use when paired with the correct data and checkpoint paths.

To finetune on OSCD dataset, the basic command is:
```shell
python train.py
```
To finetune on SegMunich dataset, the basic command is:
```shell
python -m torch.distributed.launch --nproc_per_node=2 \
--master_port=25643 --use_env train_multi_GPU_new.py
```
### Model Weights
We have already uploaded our model checkpoints [here](https://zenodo.org/records/8412455).
The [SpectralGPT.pth](https://zenodo.org/records/8412455/files/SpectralGPT.pth?download=1) checkpoint has been trained for 200 epochs on fMoW-Sentinel Dataset and the [SpectralGPT+.pth](https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1) has been continual pretrained on BigEarthNet Dataset for 100 epochs. 


Model                  |  Checkpoint 
---------------------- | -------------- 
SpectralGPT (200 epochs)  | [download](https://zenodo.org/records/8412455/files/SpectralGPT.pth?download=1)
SpectralGPT+ (100 epochs) | [download](https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1)

## Acknowledgements
Pretrain and downstream classification codes from this repository are inspired by the Masked Autoencoders (MAE) [repository](https://github.com/facebookresearch/mae) and SatMAE [repository](https://github.com/sustainlab-group/SatMAE). The downstream pixel-level codes from this repository are inspired by Seasonal Contrast (SeCo) [repository](https://github.com/ServiceNow/seasonal-contrast) and Fully Convolutional Siamese Networks for Change Detection [repository](https://github.com/rcdaudt/fully_convolutional_change_detection).

## Citation
If you found our project helpful, please kindly cite our paper:

Danfeng Hong, Bing Zhang, Xuyang Li, Yuxuan Li, Chenyu Li, Jing Yao, Naoto Yokoya, Hao Li, Xiuping Jia, Antonio Plaza, Paolo Gamba, Jon Atli Benediktsson, Jocelyn Chanussot. SpectralGPT: Spectral remote sensing foundation model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. DOI:10.1109/TPAMI.2024.3362475.

```
@article{hong2024spectralgpt,
  title={SpectralGPT: Spectral remote sensing foundation model},
  author={Hong, Danfeng and Zhang, Bing and Li, Xuyang and Li, Yuxuan and Li, Chenyu and Yao, Jing and Ghamisi, Pedram and Yokoya, Naoto and Li, Hao and Jia, Xiuping and Plaza, Antonio and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  note={DOI:10.1109/TPAMI.2024.3362475},
  year={2024}
}
```

## Licensing
Copyright (C) 2024 Danfeng Hong

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.

## Contact Information
Danfeng Hong: hongdanfeng1989@gmail.com<br>
Danfeng Hong is with the Aerospace Information Research Institute, Chinese Academy of Sciences, 100094 Beijing, China.
