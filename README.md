# GP-GS: Gaussian Processes for Enhanced 3D Gaussian Splatting

# [Paper](https://arxiv.org/pdf/2502.02283)

## 😛 Abstract
3D Gaussian Splatting has emerged as an efficient photorealistic novel view synthesis method. However, its reliance on sparse Structure-from-Motion (SfM) point clouds often limits scene reconstruction quality. To address the limitation, this paper proposes a novel 3D reconstruction framework, Gaussian Processes enhanced Gaussian Splatting (GP-GS), in which a multi-output Gaussian Process model is developed to enable adaptive and uncertainty-guided densification of sparse SfM point clouds. Specifically, we propose a dynamic sampling and filtering pipeline that adaptively expands the SfM point clouds by leveraging GP-based predictions to infer new candidate points from the input 2D pixels and depth maps. The pipeline utilizes uncertainty estimates to guide the pruning of high-variance predictions, ensuring geometric consistency and enabling the generation of dense point clouds. These densified point clouds provide high-quality initial 3D Gaussians, enhancing reconstruction performance. Extensive experiments conducted on synthetic and real-world datasets across various scales validate the effectiveness and practicality of the proposed framework.
## 😊️ Pipeline

![teaser](assets/gpgs.drawio.png)
![teaser](assets/gpgs-Page-6.drawio.png)

# GP-GS: Gaussian Processes for Enhanced Gaussian Splatting

## Overview
GP-GS is a novel framework that enhances the initialization of 3D Gaussian Splatting (3DGS) by leveraging Multi-Output Gaussian Processes (MOGP). It improves the rendering quality of novel view synthesis by densifying sparse point clouds reconstructed via Structure-from-Motion (SfM). The method is particularly effective in complex regions with densely packed objects and challenging lighting conditions.

## Pipeline
Our pipeline consists of the following steps:

1. **Multi-View Image Input**: We start with multi-view images and extract per-view depth maps using depth estimation models (e.g., Depth Anything).
2. **SfM Reconstruction**: Sparse point clouds are generated from the input images using Structure-from-Motion (SfM).
3. **Point Cloud Densification**: 
   - MOGP is trained to take pixel coordinates and depth values as input and predict dense 3D points with position and color information.
   - A Matérn kernel is used to model smooth spatial variations, and the parameters are optimized via gradient updates.
4. **Uncertainty-Based Filtering**:
   - High-variance noisy points are filtered out based on a variance-based thresholding strategy to ensure structured densification.
5. **Gaussian Initialization and Optimization**:
   - The densified points are used to initialize 3D Gaussians, which undergo further optimization to improve geometric accuracy.
6. **Novel View Rendering**:
   - The optimized 3D Gaussians are used for efficient rasterization-based rendering to synthesize high-quality novel views.
## Local setup
conda env create --file environment.yml
conda activate GP-GS
### Running
**MOGP**:
```shell
python MOGP/top_four_contribution.py #Find the image from a perspective that contributes most to SfM points cloud.
python MOGP/mogp_train.py #Training MOGP model
python MOGP/predict.py #Predict high quality dense points cloud.
```
**MOGP for 3D gaussians Initialization**:
```shell
python MOGP/rewrite_images_sfm.py
python MOGP/write_points3d.py
```
**3DGS***:
```shell
python train.py -s <scene path>
```
**Render and Evaluation**:
```shell
python render.py -m <model path>
python metrics.py -m <model path>
```


## Notes to running GaussianProcessGS  
```shell
clone repo  
conda env create --file environment.yml  
```
Changes to environment.yml:  
   Commented out # - cudatoolkit=11.6  
   Changed python version to 3.8  

   Submodeles/fused-ssim does not exist(install script expects one)  

   I found this: https://prefix.dev/channels/3dgs/packages/fused-ssim (doesn't work on Mac)  
```shell
pip install gpytorch  
pip install matplotlib  

conda activate GP-GS  
```
### This part below is for Scenes other then "flowers"

In config.py change SCENE to any dataset from mipnerf360 you want to work on  

if you want to ge Depth for datasets in mipnerf360 (other then flowers): 
```shell
   git clone https://github.com/DepthAnything/Depth-Anything-V2.git  
   pip install transformers  
   python generate_depth_map.py
```
   Output: depth folder in "scene" folder with depth map images and stacked depths  

To get the top four key frames of a scene (other then flowers):  
```shell
   python -m MOGP.top_four_contribution
``` 
   Output: mipnerf360/"scene"/top_four_images.json  

### Start here for working on "flowers":  
```shell
python -m MOGP.mogp_train (trains model and saves to gp folder in scene folder/ gives R^2 RMSE and CD)  
python -m MOGP.predict  
python -m MOGP.vis_var  
python -m rewrite_images_sfm 
python =m write_points3d (still being edited)  
```


## 📚Citation
If you find this project useful in your research, please consider cite:

```bibtex
@article{guo2025gp,
  title={GP-GS: Gaussian Processes for Enhanced Gaussian Splatting},
  author={Guo, Zhihao and Su, Jingxuan and Wang, Shenglin and Fan, Jinlong and Zhang, Jing and Han, Liangxiu and Wang, Peng},
  journal={arXiv preprint arXiv:2502.02283},
  year={2025}
}
