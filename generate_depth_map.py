import numpy as np
import os
from pathlib import Path
from colmap_io import read_points3D_text, read_extrinsics_text, read_intrinsics_text, qvec2rotmat


def generate_depth_map(scene_dir, output_path, image_name='000000.JPG'):
    images = read_extrinsics_text(os.path.join(scene_dir, "images.txt"))
    cameras = read_intrinsics_text(os.path.join(scene_dir, "cameras.txt"))
    xyzs, _, _ = read_points3D_text(os.path.join(scene_dir, "points3D.txt"))

    # Pick one image
    for img in images.values():
        if img.name == image_name:
            cam = cameras[img.camera_id]
            K = np.array([[cam.params[0], 0, cam.params[2]],
                          [0, cam.params[1], cam.params[3]],
                          [0, 0, 1]])
            
            R = qvec2rotmat(img.qvec)
            t = img.tvec
            P = K @ np.hstack((R, t.reshape(-1, 1)))

            points_h = np.hstack((xyzs, np.ones((xyzs.shape[0], 1))))
            pixels = (P @ points_h.T).T
            pixels /= pixels[:, 2][:, None]

            depths = pixels[:, 2]
            depth_map = np.zeros((cam.height, cam.width))
            for (x, y), d in zip(pixels[:, :2], depths):
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < cam.width and 0 <= yi < cam.height:
                    if depth_map[yi, xi] == 0 or d < depth_map[yi, xi]:
                        depth_map[yi, xi] = d

            np.save(output_path, depth_map)
            print(f"Saved depth map to {output_path}")
            return

    print("Image not found.")

# Example usage
if __name__ == "__main__":
    scene_folder = "mipnerf360/flowers/sparse/0"
    output_file = "mipnerf360/flowers/depth/f.npy"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # <- Add this line
    generate_depth_map(scene_folder, output_file)

