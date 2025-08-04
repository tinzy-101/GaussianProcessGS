import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
from config import SCENE_NAME, BASE_DIR
from depth_anything_v2.dpt import DepthAnythingV2
import cv2

image_dir = f"{BASE_DIR}/images"
depth_dir = f"{BASE_DIR}/depth"

# 0) Use COLMAP order if available
colmap_images = os.path.join(BASE_DIR, "sparse/0/images.txt")
if os.path.exists(colmap_images):
    order=[]
    with open(colmap_images) as f:
        for ln in f:
            ln=ln.strip()
            if not ln or ln.startswith("#"): continue
            p=ln.split()
            if p[0].isdigit() and len(p) == 10:
                order.append(p[9])  # NAME
    img_files = [os.path.join(image_dir, os.path.basename(n)) for n in order]
else:
    # Case-insensitive, stable sort
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    img_files = sorted(
        [os.path.join(image_dir,f) for f in os.listdir(image_dir)
         if os.path.splitext(f)[1] in exts],
        key=lambda s: s.lower()
    )

# 1) HF pipeline (device auto)
#pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
#DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=-1)
DEVICE = torch.device("cpu")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()



# 2) Predict, save 16-bit PNGs for visualization, stack float32 for f.npy
depths = []
H=W=None
for path in tqdm(img_files, desc="Depth-A2 (HF)"):

    raw_img = cv2.imread(path)
    D = model.infer_image(raw_img) # HxW raw depth map in numpy

    # enforce consistent size
    if H is None: H,W = D.shape
    if D.shape != (H,W):
        # resize with PIL (nearest or bilinear); choose bilinear for smoothness
        D = np.array(Image.fromarray(D).resize((W,H), resample=Image.BILINEAR), dtype=np.float32)

    depths.append(D)

    # save 16-bit PNG normalized by a robust global scale (per-image max is unstable)
    vmax = np.percentile(D, 99.9)
    if vmax <= 0: vmax = 1.0
    D16 = np.clip(D / vmax, 0, 1) * 65535.0
    D16 = D16.astype(np.uint16)

    stem = os.path.splitext(os.path.basename(path))[0]
    Image.fromarray(D16, mode="I;16").save(os.path.join(depth_dir, stem + ".png"))

F = np.stack(depths, axis=0).astype(np.float32)  # [N,H,W]
np.save(os.path.join(depth_dir, "f.npy"), F)

# record order used
with open(os.path.join(depth_dir, "image_list_colmap.txt"), "w") as g:
    g.write("\n".join([os.path.basename(p) for p in img_files]) + "\n")

print("Saved:", F.shape, "to", os.path.join(depth_dir,"f.npy"))
print("16-bit PNGs in:", depth_dir)
