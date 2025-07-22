# config.py
SCENE_NAME = "flowers"  #to train on different scene jsut change this

BASE_DIR = f"mipnerf360/{SCENE_NAME}/"
POINTS3D_PATH = BASE_DIR + "sparse/0/points3D.txt"
IMAGES_TXT_PATH = BASE_DIR + "sparse/0/images.txt"
DEPTH_FILE_PATH = BASE_DIR + "depth/f.npy"
GP_DIR = BASE_DIR + "gp/"
TEST_VAR = GP_DIR + f"{SCENE_NAME}test_var.npy"