import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import POINTS3D_PATH, IMAGES_TXT_PATH, DEPTH_FILE_PATH, BASE_DIR, TEST_VAR, PREDICT_MEAN, PIXEL_TO_POINT_POINTS3D
import os

# Add functions from mogp_train
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

def load_points3D(file_path):
    points3d_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            points3d_dict[point_id] = [x, y, z, r / 255.0, g / 255.0, b / 255.0]
    return points3d_dict

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
                       radius_factor=0.2, num_samples=10):
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

# Preprocess the track information from images.txt
def preprocess_tracks(images_txt_path):
    """Preprocess images.txt to map point3D IDs to track information."""
    tracks = {}
    with open(images_txt_path, 'r') as file:
        lines = file.readlines()
        for image_idx, line in enumerate(lines):
            if line.startswith('#') or not line.strip():
                continue
            if line.strip().endswith('.jpg') or line.strip().endswith('.png'):
                image_id = int(line.split()[0])  # IMAGE_ID is the first column
                points_line = lines[image_idx + 1].strip().split()
                for idx in range(0, len(points_line), 3):
                    x, y, p3d_id = map(float, points_line[idx:idx + 3])
                    p3d_id = int(p3d_id)
                    if p3d_id not in tracks:
                        tracks[p3d_id] = []
                    point2d_idx = idx // 3  # POINT2D_IDX is the index in the 2D points list
                    tracks[p3d_id].append((image_id, point2d_idx))
    return tracks

# Count the number of 3D points in the existing points3D.txt file
def count_3d_points(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            count += 1
    return count

# Write the points3D.txt file in bulk for better performance
def write_points3D_txt_optimized(points3D_file_path, final_points, tracks, starting_point_id):
    """Write points3D.txt in bulk for better performance."""
    point_id = starting_point_id
    lines = []

    for x, y, z, r, g, b in final_points:
        # Get TRACK[] information
        track_info = tracks.get(point_id, [])
        track_info_str = ' '.join(f"{image_id} {point2d_idx}" for image_id, point2d_idx in track_info)

        # Construct the line for this point
        line = f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.2 {track_info_str}\n"
        lines.append(line)
        point_id += 1

    # Write all lines to the file in one operation
    with open(points3D_file_path, 'a') as file:
        file.writelines(lines)


# Main script
if __name__ == "__main__":
    # Load Data and preprocess
    points3d_dict = load_points3D(POINTS3D_PATH)
    valid_data = parse_images_file(IMAGES_TXT_PATH, points3d_dict)

    # Load Depth images and normalize
    depth_images = np.load(DEPTH_FILE_PATH)
    ordered_names = load_stack_order(DEPTH_FILE_PATH, IMAGES_TXT_PATH)
    image_indices = {name: i for i, name in enumerate(ordered_names)}

    all_depths = []
    for img_name, data_points in valid_data.items():
        D = depth_images[image_indices[img_name]]     # slice for this view
        current_depth_image = collapse_depth_2d(D)      # ensure [H,W] float32
        img_height, img_width = current_depth_image.shape
        for point in data_points:
            x, y = int(point[0]), int(point[1])
            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                continue
            original_depth = current_depth_image[y, x]
            all_depths.append(original_depth)

    min_depth = np.min(all_depths)
    max_depth = np.max(all_depths)
    print(f"Depth range: {min_depth:.3f} to {max_depth:.3f}")

    # Load key four images 
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
        data_by_image_new = generate_test_data(valid_data_, DEPTH_FILE_PATH, min_depth, max_depth)

        print(data_by_image_new.keys())

        # Stack input and output data from the selected images
        all_input_data = []
        all_output_data = []

        for img_name in top_image_names:
            print(f"Inner loop: Processing image: {img_name}")
            all_input_data.append(data_by_image_new[img_name]['input'])
            all_output_data.append(data_by_image_new[img_name]['output'])

        all_input_data = np.vstack(all_input_data)
        all_output_data = np.vstack(all_output_data)

        # Normalize input and output
        input_data_normalized = all_input_data
        scaler_output = MinMaxScaler().fit(all_output_data)
        output_data_normalized = scaler_output.transform(all_output_data)

        # Load predicted data
        predicted_var = np.load(TEST_VAR)[0]
        predicted_variance = np.array(predicted_var).reshape(-1, 6)

        # Compute the 50th percentile threshold for variance
        r_var = predicted_variance[:, 3]
        g_var = predicted_variance[:, 4]
        b_var = predicted_variance[:, 5]
        rgb_mean = (r_var + g_var + b_var) / 3
        threshold = np.percentile(rgb_mean, 50)
        filtered_indices = rgb_mean <= threshold
        filtered_variance = predicted_variance[filtered_indices]

        # Load the means data
        pre_points = np.load(PREDICT_MEAN)[0]
        pre_points = np.array(pre_points).reshape(-1, 6)  # Reshape if needed, assuming the mean has the same structure as variance
        filtered_means = pre_points[filtered_indices]
        pre_points = filtered_means
        pre_points_normalized = scaler_output.inverse_transform(pre_points)

        original_pre_points = pre_points_normalized
        x, y, z = original_pre_points[:, 0], original_pre_points[:, 1], original_pre_points[:, 2]
        r = (original_pre_points[:, 3] * 255).astype(int)
        g = (original_pre_points[:, 4] * 255).astype(int)
        b = (original_pre_points[:, 5] * 255).astype(int)
        pre_final_points = np.column_stack((x, y, z, r, g, b))

        print(f"Generated {len(pre_final_points)} densified points")

        # Get starting point ID
        max_point_id = find_max_point_id(POINTS3D_PATH)
        starting_point_id = max_point_id + 1

        # Preprocess tracks
        tracks = preprocess_tracks(IMAGES_TXT_PATH)

        # Write points3D.txt to pixel-to-point directory
        write_points3D_txt_optimized(
            PIXEL_TO_POINT_POINTS3D,
            pre_final_points,
            tracks,
            starting_point_id
        )
        
        print(f"Wrote {len(pre_final_points)} new points to {PIXEL_TO_POINT_POINTS3D}")