# Data and knowledge-driven deep multiview fusion network based on diffusion model for hyperspectral image classification

## DKDMN
This work has been accepted by Expert Systems with Applications (ESWA) on March 19, 2024. At this stage, if you are interested in our work, you can debug it first and cite it when the paper is published. Be sure to read the Readme file carefully!

## Code
You can debug the code by running the **DKDMN.py** file. You will get two output features called **out_fea** and **out_cls**. The out_fea is used to calculate the contrastive loss between two views. out_cls is used to calculate loss within a view. You can construct the joint loss function from Eq.(7) to Eq.(10) in the paper.

## Data
We have provided the generative knowledge captured by the diffusion model in the Baiduyun link, which can be used directly as part of the input for the network you design.

Baiduyun Link: https://pan.baidu.com/s/160VWIPOzBFwqlyu6c0__PA

Extract Codes: HSIK

## Note
There is only **one input** to the network. You have to **concatenate** data and knowledge into an instance as the network input, and the concatenated instance will be divided into two parts for processing in the network. In addition, before concatenating the data and knowledge, they will be reduced to the **same dimension** using the PCA algorithm, respectively. The distribution knowledge of the Houston2013 dataset have been reduced to 30, and you can use the PCA algorithm to reduce the dimensionality of other datasets. Finally, for the generative knowledge code, you can look at the **Generative_Knowledge** folder and get the generative knowledge you want by running the **train_unet.py** and **feature_extract_unet.py** files. This code is also available at https://github.com/chenning0115/spectraldiff_diffusion/.

## Special Thanks!
Thanks to the SpectralDiff authors for their contributions to the hyperspectral image classification community, and welcome to cite their latest work!

N. Chen, J. Yue, L. Fang and S. Xia, "SpectralDiff: A Generative Framework for Hyperspectral Image Classification With Diffusion Models," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-16, 2023, Art no. 5522416, doi: 10.1109/TGRS.2023.3310023.
