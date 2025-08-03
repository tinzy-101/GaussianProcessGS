import os
# config.py
SCENE_NAME = "kitchen"  #to train on different scene jsut change this

BASE_DIR = f"mipnerf360/{SCENE_NAME}/"
POINTS3D_PATH = BASE_DIR + "sparse/0/points3D.txt"
IMAGES_TXT_PATH = BASE_DIR + "sparse/0/images.txt"
DEPTH_FILE_PATH = BASE_DIR + "depth/f.npy"
GP_DIR = BASE_DIR + "gp/"
TEST_VAR = GP_DIR + f"{SCENE_NAME}test_var.npy"
PREDICT_MEAN = GP_DIR + f"{SCENE_NAME}mean.npy"

# For densified results
# Use /0/ if it exists, otherwise use root
pixel_to_point_base = f"{BASE_DIR}pixel-to-point-{SCENE_NAME}"
PIXEL_TO_POINT_DIR = pixel_to_point_base + "/0/" if os.path.exists(pixel_to_point_base + "/0/") else pixel_to_point_base + "/"

PIXEL_TO_POINT_IMAGES = PIXEL_TO_POINT_DIR + "images.txt"
PIXEL_TO_POINT_POINTS3D = PIXEL_TO_POINT_DIR + "points3D.txt"