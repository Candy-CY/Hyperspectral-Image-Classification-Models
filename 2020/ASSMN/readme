# Adaptive Spectral–Spatial Multiscale Contextual Feature Extraction for Hyperspectral Image Classification (TGRS 2020)

### Di Wang, Bo Du, Liangpei Zhang and Yonghao Xu

### Update 2021.07: ASSMN won the Highly Cited Paper.

![](https://github.com/DotWang/ASSMN/blob/master/highcited.png)


## Framework

![](https://github.com/DotWang/ASSMN/blob/master/model.png)

## Usage (Pytorch implementation)

1. Install Pytorch 1.1 with Python 3.5.

2. Clone this repo.

```
git clone https://github.com/DotWang/ASSMN.git
```

3. Training and evaluation with ***trainval.py***.

      For example, for [Indian Pines dataset](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), if SeMN and SaMN are all employed:

```
CUDA_VISIBLE_DEVICES=0 python trainval.py \
	--dataset 'indian' \
	--dr-num 4 --dr-method 'pca' \
	--mi -1 --ma 1 \
	--half-size 13 --rsz 27 \
	--experiment-num 10 \
	--lr 1e-2 --epochs 200 --batch-size 16 \
	--scheme 2 --strategy 's2' \
	--spec-time-steps 2 \
	--group 'alternate' --seq 'cascade' \
	--npi-num 2
```


&emsp; &ensp; Then the assessment results are recorded in the corresponding ***\*.mat*** file and the generated model is saved.


4.  Predicting with the previous stored model through ***infer.py***

```
CUDA_VISIBLE_DEVICES=0 python infer.py \
      --dataset 'indian' \
      --mi -1 --ma 1 \
      --half-size 13 --rsz 27 \
      --bz 50000 \
      --scheme 2 --strategy 's2' 
```
&emsp; &ensp; and then produce the final classification map.

## Paper and Citation

If this repo is useful for your research, please cite our [paper](https://doi.org/10.1109/TGRS.2020.2999957).

```
@ARTICLE{wd_2021_assmn,
  author={D. {Wang} and B. {Du} and L. {Zhang} and Y. {Xu}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Adaptive Spectral–Spatial Multiscale Contextual Feature Extraction for Hyperspectral Image Classification}, 
  year={2021},
  volume={59},
  number={3},
  pages={2461-2477},
  doi={10.1109/TGRS.2020.2999957}
  }
```

## Acknowledgement
Thanks [Andrea Palazzi](https://github.com/ndrplz/ConvLSTM_pytorch) for providing the Pytorch implementation of Convolutional LSTM!

## Relevant Projects
[1] <strong> Image-level/Patch-free Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Fully Contextual Network for Hyperspectral Scene Parsing, IEEE TGRS, 2021 | [Paper](https://ieeexplore.ieee.org/document/9347487) | [Github](https://github.com/DotWang/FullyContNet)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, and Liangpei Zhang</em>

[2] <strong> Graph Convolution based Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Spectral-Spatial Global Graph Reasoning for Hyperspectral Image Classification, IEEE TNNLS, 2023 | [Paper](https://arxiv.org/abs/2106.13952) | [Github](https://github.com/DotWang/SSGRN)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, and Liangpei Zhang</em>
 
[3] <strong> Neural Architecture Search for Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; HKNAS: Classification of Hyperspectral Imagery Based on Hyper Kernel Neural Architecture Search, IEEE TNNLS, 2023 | Paper | [Github](https://github.com/DotWang/HKNAS)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, Liangpei Zhang, and Dacheng Tao</em>

[4] <strong> ImageNet Pretraining and Transformer based Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; DCN-T: Dual Context Network with Transformer for Hyperspectral Image Classification, IEEE TIP, 2023 | [Paper](https://arxiv.org/abs/2304.09915) | [Github](https://github.com/DotWang/DCN-T)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Jing Zhang, Bo Du, Liangpei Zhang, and Dacheng Tao</em>
