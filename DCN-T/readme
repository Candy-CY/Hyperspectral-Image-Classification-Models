# DCN-T: Dual Context Network with Transformer for Hyperspectral Image Classification (TIP 2023)

## Di Wang, Jing Zhang, Bo Du, Liangpei Zhang and Dacheng Tao

### Pytorch implementation of our [paper](https://arxiv.org/abs/2304.09915) for ImageNet Pretraining and Transformer-based image-level hyperspectral image classification.


<table>
<tr>
<td><img src=Figs/network.png width=500>
<br> 
<figcaption align = "left"><b>Fig.1 - The proposed DCN-T. </b></figcaption></td>
<td><img src=Figs/module.png width=355>
<br> 
<figcaption align = "right"><b>Fig.2 - The DCM. </b></figcaption> </td>
</tr>
</table>

## Usage
1. Install Pytorch 1.9 with Python 3.8.
2. Clone this repo.
```
git clone https://github.com/DotWang/DCN-T.git
```
3. Prepare the tri-spectral dataset with the notebook
4. Download [ImageNet pretrained model](https://pytorch.org/vision/stable/models/vgg.html?highlight=models)
5. For implementing the clusttering, install the SSN 
```
cd utils/gensp/src
python setup.py install
```
Or you can taste the pytorch version realized in the ***network_local_global.py***

6. Training and Testing

For example, training on the [WHU-Hi-LongKou](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) scene with soft voting

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes 1 \
    --node_rank=0 --master_port=1901 --use_env train_memory.py \
    --dataset 'WHUHi_LongKou_15_100' \
    --backbone 'vgg16' \
    --epochs 30 --lr 1e-3 --groups 128 --eval_interval 1 \
    --batch_size 4 --test_batch_size 1 --workers 2 \
    --ra_head_num 4 --ga_head_num 4 --mode 'soft'
```

```
CUDA_VISIBLE_DEVICES=0 python test_gpu.py \
    --dataset 'WHUHi_LongKou_15_100' \
    --backbone 'vgg16' --ra_head_num 4 --ga_head_num 4 \
    --scales 1  --groups 128 \
    --model_path './run/WHUHi_LongKou_15_100/vgg16_128/experiment_0/model_last.pth.tar' \
    --save_folder './run/WHUHi_LongKou_15_100/vgg16_128/experiment_0/'
```


## Citation

```
@ARTICLE{wang_2023_dcnt,
  author={Wang, Di and Zhang, Jing and Du, Bo and Zhang, Liangpei and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing}, 
  title={DCN-T: Dual Context Network with Transformer for Hyperspectral Image Classification}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2023.3270104}}
```

## Thanks
[SSN-Pytorch](https://github.com/perrying/ssn-pytorch) &ensp; [SpixelFCN](https://github.com/fuy34/superpixel_fcn) &ensp; [BIT](https://github.com/justchenhao/BIT_CD)


## Relevant Projects
[1] <strong> Pixel and Patch-level Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Adaptive Spectral–Spatial Multiscale Contextual Feature Extraction for Hyperspectral Image Classification, IEEE TGRS, 2020 | [Paper](https://ieeexplore.ieee.org/document/9121743/) | [Github](https://github.com/DotWang/ASSMN)
<br> <em> &ensp; &ensp;  Di Wang<sup>&#8727;</sup>, Bo Du, Liangpei Zhang and Yonghao Xu</em>

[2] <strong> Image-level/Patch-free Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Fully Contextual Network for Hyperspectral Scene Parsing, IEEE TGRS, 2021 | [Paper](https://ieeexplore.ieee.org/document/9347487) | [Github](https://github.com/DotWang/FullyContNet)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, and Liangpei Zhang</em>
 
[3] <strong> Graph Convolution based Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Spectral-Spatial Global Graph Reasoning for Hyperspectral Image Classification, IEEE TNNLS, 2023 | [Paper](https://arxiv.org/abs/2106.13952) | [Github](https://github.com/DotWang/SSGRN)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, and Liangpei Zhang</em>
 
[4] <strong> Neural Architecture Search for Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; HKNAS: Classification of Hyperspectral Imagery Based on Hyper Kernel Neural Architecture Search, IEEE TNNLS, 2023 | Paper | [Github](https://github.com/DotWang/HKNAS)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, Liangpei Zhang, and Dacheng Tao</em>
