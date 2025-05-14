# Can Spectral Information Work While Extracting Spatial Distribution? — An Online Spectral Information Compensation Network for HSI Classification
Tensorflow implementation of OSICN for hyperspectral image classification.
# Installation
Install Tensorflow 2.4.0 with Python 3.8.
# Usage
The HSI datasets used in this psper can be downloaded from the following websites:  

https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes  
http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm  
https://hyperspectral.ee.uh.edu/?page_id=1075
  
  
Please first change the path in the code, and then run ('GPU_Number' can be selected as '0, 1, 2...'):  
  
CUDA_VISIBLE_DEVICES=GPU_Number python ./OSICN.py  
  
  
Please note that the results may be varied due to differences in the training of the network.
# Paper
Can Spectral Information Work While Extracting Spatial Distribution? — An Online Spectral Information Compensation Network for HSI Classification  
  
Please kindly cite our paper if you find it's helpful for your work. Thank you!
