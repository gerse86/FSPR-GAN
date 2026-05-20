# FSPR-GAN
This repository is the official PyTorch implementation of the paper "FSPR-GAN: A StyleGAN2-Based Framework for Few-Shot 3D Reconstruction of Porous Media", which addresses the issues of discriminator overfitting, unstable training and mode collapse in traditional generative adversarial networks under limited-data conditions, and can generate high-fidelity 128<sup>3</sup> voxel porous media 3D models.
# Authors
Yang Chen<sup>a,b</sup>, Wuwen Yao<sup>a,b,c,[]</sup>, Xin Liu<sup>a,b</sup>, Jirong Yi<sup>d</sup>, Chuang Deng<sup>c</sup>
<sup>a</sup> Key Laboratory of Safety Control of Bridge Engineering, Ministry of Education, Changsha University of Science and Technology, Changsha, 410114, Hunan, China 
<sup>b</sup> School of Civil and Environmental Engineering, Changsha University of Science and Technology, Changsha, 410114, China 
<sup>c</sup> Department of Mechanical Engineering, University of Manitoba, Winnipeg, R3T 5V6, Manitoba, Canada
<sup>d</sup> Artificial Intelligence in Medicine, Cedars-Sinai Medical Center, Los Angeles, CA 90007, USA
# Core code structure
augment.py: 3D adaptive data augmentation pipeline
network.py: Core architecture of generator and discriminator
train.py: Main script for model training
predict.py: Model inference and sample generation
Metric.py: Metrics for quantitative evaluation of pore structure
MetricChecker.py: Automatic metric verification during training
utils.py: Utility functions and loss function definitions
# Environmental requirements
Python 3.9+
PyTorch 2.0+ (CUDA 11.8+)
Dependencies: See requirements.txt file
# Data Preparation
The dataset is sourced from the Digital Porous Media Portal and has been preprocessed into a 128×128×128 binary .raw format (0 = pore, 1 = matrix).
# Training and Inference
Training: After modifying the dataset path in train.py, run "python train.py" to support resuming training from a breakpoint.
Inference: After modifying the checkpoint path in predict.py, run "python predict.py" to generate 3D volume data.
# Experimental results
On the Bentheimer and Castlegate sandstone dataset, the generated samples were highly consistent with the real samples in terms of porosity, connectivity, and the distribution of pore - throat sizes, among other metrics.

