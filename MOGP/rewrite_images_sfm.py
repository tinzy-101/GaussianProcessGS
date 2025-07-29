import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
from config import POINTS3D_PATH, IMAGES_TXT_PATH, DEPTH_FILE_PATH, BASE_DIR, TEST_VAR,  PREDICT_MEAN,PIXEL_TO_POINT_IMAGES


file_path_points3d = POINTS3D_PATH
file_path_images = IMAGES_TXT_PATH
depth_file_path = DEPTH_FILE_PATH

#Constants
RADIUS = 0.4
DYNAMIC_MVNTS = 10
THRESHOLD = 50

#Add functions from mogp_train
def load_image_order_from_txt(images_txt_path):
    order = []
    with open(images_txt_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'): 
                continue
            p = ln.split()
            if p[0].isdigit() and len(p) >= 10:
                order.append(p[9])
    return order

def load_stack_order(depth_file_path, fallback_images_txt):
    # Prefer the explicit list you wrote when stacking f.npy
    dir_depth = os.path.dirname(depth_file_path)
    order_file = os.path.join(dir_depth, "image_list_colmap.txt")
    if os.path.exists(order_file):
        with open(order_file) as f:
            return [ln.strip() for ln in f if ln.strip()]
    # Fallback: parse images.txt (in-file order)
    return load_image_order_from_txt(fallback_images_txt)


def collapse_depth_2d(D):
    """D: [H,W] or [H,W,C] -> [H,W] float32."""
    if D.ndim == 2:
        return D.astype(np.float32)
    # use luminance of first 3 channels; ignore alpha if present
    if D.shape[2] >= 3:
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return (D[..., :3].astype(np.float32) @ w).astype(np.float32)
    return D[..., 0].astype(np.float32)


def find_max_point_id(file_path):
    max_point_id = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            point_id = int(parts[0])
            max_point_id = max(max_point_id, point_id)
    return max_point_id


def count_3d_points(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comment lines and empty lines
            if line.startswith('#') or not line.strip():
                continue
            count += 1
    return count


num_3d_points = count_3d_points(file_path_points3d)
print("Number of 3D points:", num_3d_points)

max_point_id = find_max_point_id(file_path_points3d)
print("Maximum point ID:", max_point_id)

def load_points3D(file_path):
    points3d_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            #r, g, b = map(int, parts[4:7])
            r, g, b = map(lambda v: int(float(v)), parts[4:7])

            points3d_dict[point_id] = [x, y, z, r / 255.0, g / 255.0, b / 255.0]
    return points3d_dict


def update_images_txt_optimized(images_txt_path, predictions):
    # Read the entire file content
    with open(images_txt_path, 'r') as file:
        lines = file.readlines()

    image_indices = {}
    for i, line in enumerate(lines):
        match = re.match(r"^\d+\s.*\s(?P<name>.+\.(jpg|png))$", line)
        if match:
            image_name = match.group("name")
            image_indices[image_name] = i

    # Construct updated lines in memory
    for image_name, points in predictions.items():
        if image_name in image_indices:
            points_line_index = image_indices[image_name] + 1
            points_line = lines[points_line_index].strip()
            points_line += ''.join(f" {x} {y} {points3D_ID}" for x, y, points3D_ID in points)
            lines[points_line_index] = points_line + "\n"

    # Write all lines back to the file in one operation
    with open(images_txt_path, 'w') as file:
        file.writelines(lines)


def parse_images_file(file_path, points3d_dict):
    valid_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('#') or not lines[i].strip():
                i += 1
                continue

            image_data = lines[i].strip().split()
            image_name = image_data[9]
            i += 1
            keypoints_data = lines[i].strip().split()
            points2d = []
            k = 0
            while k < len(keypoints_data):
                x, y = map(float, keypoints_data[k:k+2])
                point3d_id = int(keypoints_data[k+2])
                if point3d_id != -1 and point3d_id in points3d_dict:
                    points2d.append((x, y) + tuple(points3d_dict[point3d_id]))
                k += 3
            if points2d:
                valid_data[image_name] = points2d
            i += 1

    return valid_data


def generate_test_data(valid_data, depth_file_path, min_depth, max_depth,
                       radius_factor=RADIUS, num_samples=DYNAMIC_MVNTS):
    depth_images = np.load(depth_file_path)                # [N,H,W] or [N,H,W,C]
    print("depth stack shape:", depth_images.shape)

    ordered_names = load_stack_order(depth_file_path, IMAGES_TXT_PATH)
    image_indices = {name: i for i, name in enumerate(ordered_names)}

    data_by_image = {}

    for image_name, data_points in valid_data.items():
        if image_name not in image_indices:
            # name mismatch (e.g., case/relative path) → skip safely
            continue

        D = depth_images[image_indices[image_name]]
        # Collapse multi-channel → single channel float32
        if D.ndim == 3:
            h, w, c = D.shape
            if c >= 3:
                weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                D = (D[..., :3].astype(np.float32) @ weights).astype(np.float32)
            else:
                D = D[..., 0].astype(np.float32)
        else:
            D = D.astype(np.float32)

        H, W = D.shape
        # dynamic radius and movement set
        r = int(radius_factor * min(H, W))
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        movements = [(int(r*np.cos(a)), int(r*np.sin(a))) for a in angles]

        input_data, output_data, test_data = [], [], []
        for p in data_points:
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < W and 0 <= y < H):
                continue
            d = float(D[y, x])
            # normalize depth globally (same min/max you computed)
            nd = (d - min_depth) / (max_depth - min_depth + 1e-12)
            input_data.append([x / W, y / H, nd])      # normalized here
            output_data.append(p[2:])                  # (X,Y,Z,R,G,B)

            # test samples (not used for ranking, but keep correct indexing)
            for dx, dy in movements:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    nd2 = (float(D[ny, nx]) - min_depth) / (max_depth - min_depth + 1e-12)
                    test_data.append([nx / W, ny / H, nd2])

        data_by_image[image_name] = {
            'input':  np.asarray(input_data,  dtype=float),   # (N,3)
            'output': np.asarray(output_data, dtype=float),   # (N,6)
            'test':   np.asarray(test_data,   dtype=float),   # (M,3)
        }
    return data_by_image


def recover_test_data(test_data_normalized, image_width, image_height, starting_point_id):
    # Revert normalization for x and y
    test_data_recovered = test_data_normalized.copy()
    test_data_recovered[:, 0] *= image_width  # Denormalize x
    test_data_recovered[:, 1] *= image_height  # Denormalize y
    test_data_recovered[:, 0] = np.round(test_data_recovered[:, 0]).astype(int)
    test_data_recovered[:, 1] = np.round(test_data_recovered[:, 1]).astype(int)

    # points3d_id from start_point to end_point(shape-1)
    point_ids = np.arange(starting_point_id, starting_point_id + test_data_recovered.shape[0])

    # Combine x, y, and point IDs
    recovered_data = np.column_stack((
        test_data_recovered[:, 0],  # x
        test_data_recovered[:, 1],  # y
        point_ids                  # points3D_ID
    ))

    return recovered_data

#Load Data and preprocess
points3d_dict = load_points3D(file_path_points3d)
valid_data = parse_images_file(file_path_images, points3d_dict)

#Load Depth images and normalize
depth_images = np.load(depth_file_path)
ordered_names = load_stack_order(depth_file_path, file_path_images)
image_indices = {name: i for i, name in enumerate(ordered_names)}

all_depths =[]

for img_name, data_points in valid_data.items():
    D = depth_images[image_indices[img_name]]     # slice for this view
    current_depth_image = collapse_depth_2d(D)      # ensure [H,W] float32
    img_height, img_width = current_depth_image.shape
    for point in data_points:
        x,y = int(point[0]), int(point[1])
        if x < 0 or x >= img_width or y<0 or y>= img_height:
            continue
        original_depth = current_depth_image[y,x]
        all_depths.append(original_depth)

min_depth = np.min(all_depths)
max_depth = np.max(all_depths) 
print(f"Depth range: {min_depth:.3f} to {max_depth:.3f}")



#Load key four images 
# top_images_path = os.path.join(BASE_DIR,"top_four_images.json")
# with open(top_images_path, "r") as f:
#     top_image_names = json.load(f)

#images = ["000115.JPG", "000101.JPG", "000079.JPG", "000072.JPG"]
images = ["000115.JPG"]

for image_name in images:
    print(f"Processing image: {image_name}")
    top_image_names = [image_name]

    print(f"Selected images: {top_image_names}")

    #filter on the top 4 images
    valid_data_ = {k: v for k, v in valid_data.items() if k in top_image_names}

    #Generate input/ouput/test sets
    data_by_image_new = generate_test_data(valid_data_, depth_file_path, min_depth,max_depth)

# Process all key images using the scene-level prediction files
predictions = {}
current_point_id = max_point_id + 1

# Load predicted variance and mean for the entire scene
predicted_var = np.load(TEST_VAR)[0]
predicted_variance = np.array(predicted_var).reshape(-1, 6)

# Filter by uncertainty
r_var = predicted_variance[:, 3]
g_var = predicted_variance[:, 4]
b_var = predicted_variance[:, 5]
rgb_mean = (r_var + g_var + b_var) / 3
threshold = np.percentile(rgb_mean, THRESHOLD)
filtered_indices = rgb_mean <= threshold

print(f"Total predicted points: {len(predicted_variance)}")
print(f"Points after filtering: {np.sum(filtered_indices)}")

# Process each key image
for image_name in data_by_image_new.keys():
    print(f"Processing image: {image_name}")
    
    # Get test data for this image
    test_data_normalized = data_by_image_new[image_name]['test']
    
    # Apply the same filtering to this image's test data
    # (assuming the test data order matches the prediction order)
    if len(test_data_normalized) > 0:
        # Get the corresponding slice of filtered indices for this image
        # This assumes test data is processed in the same order as predictions
        start_idx = sum(len(data_by_image_new[img]['test']) for img in sorted(data_by_image_new.keys()) if img < image_name)
        end_idx = start_idx + len(test_data_normalized)
        
        # Get filtered test data for this image
        image_filtered_indices = filtered_indices[start_idx:end_idx]
        test_data_filtered = test_data_normalized[image_filtered_indices]
        
        print(f"  Original test points: {len(test_data_normalized)}")
        print(f"  Filtered test points: {len(test_data_filtered)}")
        
        if len(test_data_filtered) == 0:
            print(f"  No points remaining after filtering for {image_name}, skipping...")
            continue
        
       
        # Get image dimensions for denormalization
        image_idx = image_indices[image_name]
        D = depth_images[image_idx]
        current_depth_image = collapse_depth_2d(D)
        image_height, image_width = current_depth_image.shape
        
        
        # Recover test data (denormalize and assign point IDs)
        test_data_recovered = recover_test_data(
            test_data_filtered,
            image_width,
            image_height,
            current_point_id
        )
        test_data_recovered = test_data_recovered.astype(int)
        
        # Add to predictions dictionary
        predictions[image_name] = [(int(row[0]), int(row[1]), int(row[2])) for row in test_data_recovered]
        
        # Update point ID counter
        current_point_id += len(test_data_recovered)
        
        print(f"  Added {len(test_data_recovered)} new correspondences for {image_name}")
    else:
        print(f"  No test data for {image_name}, skipping...")

# Update pixel-to-point/images.txt with all new correspondences
images_txt_path =PIXEL_TO_POINT_IMAGES


if predictions:
    update_images_txt_optimized(images_txt_path, predictions)
    print(f"Updated {len(predictions)} images with new correspondences")
else:
    print("No predictions to write - all images were skipped or had no valid points after filtering")