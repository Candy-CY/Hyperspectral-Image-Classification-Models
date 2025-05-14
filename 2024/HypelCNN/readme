# HypeLCNN Overview

This repository includes source codes for hyperspectral and LiDAR fusion system for classification and GAN based
hyperspectral sample generation.

This repository includes an integrated suite for hyperspectral and lidar, neural net based classification.

Primary features are:

- Plug-in based model implementation( via NNModel interface )
- Plug-in based data set integration( via DataLoader interface )
- Data efficient implementations for training( memory efficient/in-memory/record based )
- Cross use of data set integration in classic machine learning and deep learning methods
- Integrated hyperparameter optimization (via optuna).
- Training, classification and metrics integration for neural networks
- CPU/GPU based training
- Sample implementations for 
  - CNN networks
  - Capsule networks
- GAN based data augmenter implementation with training and evaluation codes.
  - Vanilla GAN
  - Cycle GAN
  - Contrastive Unpaired Translation(CUT) GAN
  - Dual Contrastive Learning Adversarial Generative Networks (DCLGAN)

Project is developed using Tensorflow (Tested on version 2.9.0). For now, it works on TF2 using legacy mode. 
Eager mode will be implemented in the upcoming days.

For Tensorflow 1.x version, please use branch "tf1_support"(tested on Tensorflow 1.15). 
Source codes in this branch can be used for best practices of applying Tensorflow 1.x for;

- Training large data sets
- Merging two different neural networks for data augmentation
- Integrating custom metrics
- Reading summary file and extracting extra information from it.

# Data sets

Data set files are too large to commit on repository. Access to data sets can be done using this URL;

https://drive.google.com/drive/folders/1oiNRuPAkQ1MpJEUjKYE8YtXpcM405usE?usp=sharing

Each data set has its own licensing, and you should include them in your paper/work.

# Requirements

Works with Python 3.7-3.10. Library dependencies are put in "requirements.txt".

# Source Files

- nnmodel package => Classification model abstraction and implementation
  - NNModel.py : Neural Network definition interface, provides methods for declaring loss function, base parameter
    values, hyperparameter value range.
  - HYPELCNNModel.py : Model declaration source code for HypeLCNN.
  - DUALCNNModel.py : Model declaration source code for CNN Dual Model.
  - CONCNNModel.py : Model declaration source code for Context CNN Model.
  - CAPModel.py : Capsule Network Model.
  - modelconfigs => Parameter value configs for corresponding NN models.
- importer package => TF import strategies
  - DataImporter.py : DataImporter interface class.
  - InMemoryImporter.py : Placeholder based in memory data importer(fastest but memory inefficient
    implementation).
  - GeneratorImporter.py : Generator based data importer(memory efficient but slower than in memory
    implementation.
  - TFRecordImporter.py : TFRecord based data importer. TFRecord files can be created using
    tfrecord_writer in utilities package.
- loader package => Data set loader abstraction and various implementations.
  - DataLoader.py : DataLoader interface class, used for integrating different data sets to the training/inference
    system.
  - GRSS2013DataLoader.py : GRSS2013 data set loader implementation.
  - GRSS2018DataLoader.py : GRSS2018 data set loader implementation.
  - GULFPORTDataLoader.py : Gulfport data set loader implementation.
  - GULFPORTALTDataLoader.py : Gulfport data set alternative loader implementation(shadow data augmentation).
  - AVONDataLoader.py : AVON data set loader implementation.
- gan package => HSI shadow data transformation learning and shadowed data sampling with various GAN methods.
  - gan/gan_infer_for_shadow.py : Gan inference implementation.
  - gan/gan_train_for_shadow.py : Gan training runner.
  - gan.wrapper package
      - gan/*_wrapper.py : Wrappers for various gan methods.
- utilities package => Various utility functions for HSI visualization, TF summary read, HSI-LIDAR registration 
  - tfrecord_writer.py : TF record generator from samples.
  - hsi_rgb_converter.py : HSI to RGB conversion script.
  - read_summary_file.py : Reads tensorflow summary file and extract some statistical information.
- classify package => Training and inference apps for classification. 
  - train_for_classification.py : Terrain classification training execution class.
  - infer_for_classification.py : Loads a checkpoint file and data set and performs scene classification.
  - monitored_session_runner.py : Tensorflow(1.x) Monitored session runner implementation.
  - classic_ml_trainer.py : SVM and Random Forest Classifier implementation using scikit-learn.
- common package => Common libraries for other packages.
  - common_nn_operations.py : Common nn methods.
  - cmd_parser.py : Common cmd parsers and flags.
- notebook.ipynb : Sample notebook for executing the developed ML models.

# In progress

- Source code documentation.
- TF 2.x Eager Mode transition.

# Citation
If you use this in your research, please cite the paper:

```bibtex
@article{peker2024shadow,
  title={Shadow-aware terrain classification: advancing hyperspectral image sensing through generative adversarial networks and correlated sample synthesis},
  author={Peker, Ali Gokalp and Yuksel, Seniha Esen and Cinbis, Ramazan Gokberk and Cetin, Yasemin Yardimci},
  journal={Journal of Applied Remote Sensing},
  volume={18},
  number={3},
  pages={038506--038506},
  year={2024},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}
```
- Pytorch implementation.
