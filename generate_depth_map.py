import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
from config import SCENE_NAME, BASE_DIR

#Configurations

image_dir = f"{BASE_DIR}/images"
depth_dir = f"{BASE_DIR}/depth"
output_npy_path = os.path.join(depth_dir, "f.npy")
depth_model = "depth-anything/Depth-Anything-V2-Large-hf"

#Set-up the pipeline
pipe = pipeline(tasks="depth-estimation", model=depth_model)

# create directories for new depth files
os.makedirs(depth_dir, exist_ok = True)

#Load Images
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
depths = []

for fname in tqdm(image_files, desc="Generating depth maps"):
    img_path = os.path.join(image_dir, fname)
    image = Image.open(img_path).convert("RGB")

    #Predict depth
    result = pipe(image)
    depth = result["depth"] #NumPy array
    depths.append(depth)

    #save greyscale depth images for visual in depth folder
    depth_np = np.array(depth)
    depth_normalized = (depth_np / depth_np.max()* 255).astype(np.uint8)
    depth_image_path = os.path.join(depth_dir, fname.replace('.JPG','.png'))
    Image.fromarray(depth_normalized).save(depth_image_path)


#saving the stacked depth array
depth_array = np.stack(depths, axis=0)
np.save(output_npy_path,depth_array)

print(f"\n Depth maps saved to {depth_dir}")
print(f"Depth array saves as:{output_npy_path}")
print (f"Shape: {depth_array.shape} (num_images, height, width)")