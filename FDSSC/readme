
# A Fast Dense Spectral-Spatial Convolution Network Framework for Hyperspectral Images Classification
This is the Keras implementation of the [paper](http://www.mdpi.com/2072-4292/10/7/1068) (accpted by Remote Sensing in 3 July 2018). More information about the paper is in [here](https://shuguang-52.github.io/fdssc/) 

<img src='figures/FDSSC Network.PNG'>

**Fig 1**.FDSSC network

## Citation
If you find FDSSC useful in your research, please consider citing.

**Chicago/Turabian Style**:

Wang, Wenju; Dou, Shuguang; Jiang, Zhongmin; Sun, Liujie.	2018. "A Fast Dense Spectral–Spatial Convolution Network Framework for Hyperspectral Images Classification." Remote Sens. 10, no. 7: 1068.

## Setup
### **Python 3.5.x**
 
You can download Python 3.5.3 at [here](https://www.python.org/downloads/release/python-353/).

### **Tensorflow-gpu v1.8.0**
 
We use Tensorflow-gpu as our computing backend, and you can also use [theano](http://deeplearning.net/software/theano/install.html) as computing backend. For downloading tensorflow, you can find all the information you need at [here](https://www.tensorflow.org/install/). 
If you have pip, you can simply input the following command:

    pip install --user tensorflow-gpu==1.8.0
    
### **CUDA v9.0 and cuDNN v7.0**
To run the tensorflow-gpu, you need a Nvidia GPU card with CUDA Compute Capability 3.0 or higher. 

For our source codes, it would be best that you have a CUDA enabled GPU with you, that would save you a lot of time. But it is still OK to run all the experiments without GPU, at the expense of more patience. For example, for the KSC dateset, we used GTX 1080Ti (11GB, Compute Capability = 6.1) to train our model within about 201.2 seconds.  

+ You can download CUDA v9.0 at [here](https://developer.nvidia.com/cuda-90-download-archive). Ensure that you append the relevant Cuda pathnames to the %PATH% environment variable as described in the NVIDIA documentation.

+ You can dowmload cuDNN v7.0 at [here](https://developer.nvidia.com/rdp/cudnn-archive). Note that cuDNN is typically installed in a different location from the other CUDA DLLs. Ensure that you add the directory where you installed the cuDNN DLL to your %PATH% environment variable.

If you have a different version of one of the preceding packages, please change to the specified versions. In particular, the cuDNN version must match exactly: TensorFlow will not load if it cannot find cuDNN64_7.dll. To use a different version of cuDNN, you must build from source.

### **Keras 2.1.6**
To install it and related development package, type:

    pip install --user numpy scipy matplotlib scikit-learn scikit-image
    pip install --user keras==2.1.6


### **Model visualization**
The keras.utils.vis_utils provides a function to draw the Keras model (using graphviz) and we used it in our code. It relies on pydot-ng and graphviz. To install it, type:

     pip install pydot-ng & brew install graphviz
   
Or you can delete related code. It doesn't matter with classification results.
## Dataset
You can get all HSI datasets by running download_datasets.py.
Or you can download IN, KSC and UP dataset at [here](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)

## Reproducing the results
1) Run the "train.py". You need type the name of HSI dataset. Model are saved with a hdf5 format in ./models file.

2) Run the "evaluate_model.py", in order to evaluate saved models. You need type the name of HSI dataset and the time series number of model(such as '04\_17\_14\_36').

3) Run the "get\_classification\_maps.py", for creating the clasification map. You also need type the name of HSI dataset and the time series number of model. And you will get the result in .mat format and classification maps.

## Experiment results
<img src="figures/KSC.PNG">

**Fig 2**.Classification maps for KSC dataset: (a) real image of one band in KSC dataset; (b) ground-truth map; (c) SAE-LR, OA = 92.99%; (d) CNN, OA = 99.31%; (e) SSRN, OA = 99.94%; and (f) FDSSC, OA = 99.96%.

<img src="figures/UP.PNG">

**Fig 3**.Classificatin maps for Figure 11 Classification maps for UP dataset: (a) real image of one band in UP dataset; (b) ground-truth map; (c) SAE-LR, OA = 98.46%; (d) CNN, OA = 99.38%; (e) SSRN, OA = 99.93%; and (f) FDSSC, OA = 99.96%.

<img src="figures/IN.PNG">

**Fig 4**.Classification maps for IN dataset: (a) real image of one band in the IN dataset; (b) ground-truth map; (c) SAE-LR, OA = 93.98%; (d) CNN, OA = 95.96%; (e) SSRN, OA = 99.35%; and (f) FDSSC, OA = 99.72%
