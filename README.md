
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
