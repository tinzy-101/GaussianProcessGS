import torch

DTYPE = torch.float64

torch.set_default_dtype(DTYPE)

import os
import json
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  # Import tqdm for progress bar


from config import (
    POINTS3D_PATH, 
    DEPTH_FILE_PATH, 
    IMAGES_TXT_PATH,
    BASE_DIR,
    SCENE_NAME
)
RADIUS = 0.25
DYNAMIC_MVNTS = 8
TEST_SIZE = 0.2
RANDOM_STATE = 42
TRAINING_ITERATIONS = 1000
NUM_TASKS = 6
LEARNING_RATE = 0.2
WEIGHT_DECAY =1e-6 
NU = 0.5

#Set up output directory if it doesn't exist
GP_OUTPUT_DIR = os.path.join(BASE_DIR, "gp")
os.makedirs(GP_OUTPUT_DIR, exist_ok= True)


# Prepare data paths
file_path_points3d = POINTS3D_PATH
depth_file_path = DEPTH_FILE_PATH
file_path_images = IMAGES_TXT_PATH



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

# Load points3D from file
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

# Parse images file: matches image pixel coordinated to corresponding 3d points
#uses Colmap's images.txt
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


#Load and Preprocess data
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

    # Split data into train and test sets
    train_input, test_input, train_output, test_output = train_test_split(input_data_normalized, output_data_normalized, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_input = torch.tensor(train_input, dtype=DTYPE)
    train_output = torch.tensor(train_output, dtype=DTYPE)
    test_input = torch.tensor(test_input, dtype=DTYPE)
    test_output = torch.tensor(test_output, dtype=DTYPE)

    import pandas as pd
    # train_x_df = pd.DataFrame(train_input)
    # print(train_x_df.describe())
    # train_y_df = pd.DataFrame(train_output)
    # print(train_y_df.describe())

    class MultiTaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, num_tasks):
            super(MultiTaskGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean(batch_shape=torch.Size([num_tasks]))

            self.covar_module = ScaleKernel(
                MaternKernel(nu=NU, batch_shape=torch.Size([num_tasks])),
                batch_shape=torch.Size([num_tasks])
            )




        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(mean_x, covar_x))


    def chamfer_distance(pred_points, true_points):
        pred_expanded = pred_points.unsqueeze(1)  # (N, 1, D)
        true_expanded = true_points.unsqueeze(0)  # (1, M, D)

        # Pairwise distances between predicted points and true points
        distances = torch.norm(pred_expanded - true_expanded, dim=-1)  # (N, M)

        # Compute Chamfer Distance
        forward_cd = torch.mean(torch.min(distances, dim=1)[0])
        backward_cd = torch.mean(torch.min(distances, dim=0)[0])

        return forward_cd + backward_cd


    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    likelihood = MultitaskGaussianLikelihood(num_tasks=6).to(device)
    model = MultiTaskGPModel(train_input.to(device), train_output.to(device), likelihood, num_tasks=NUM_TASKS).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE)#, weight_decay=WEIGHT_DECAY)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    model.train()
    likelihood.train()
    for i in tqdm(range(TRAINING_ITERATIONS)):
        optimizer.zero_grad()
        output = model(train_input.to(device))
        loss = -mll(output, train_output.to(device))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        #print(f'Iteration {i + 1}/{TRAINING_ITERATIONS}, Loss: {loss.item()}')


    #Save the output
    torch.save(model.state_dict(), os.path.join(GP_OUTPUT_DIR, f"{SCENE_NAME}.pth"))
    torch.save(likelihood.state_dict(), os.path.join(GP_OUTPUT_DIR, f"{SCENE_NAME}likelihood.pth"))
    np.save(os.path.join(GP_OUTPUT_DIR, f"{SCENE_NAME}.npy"), np.array(losses))


    # Evaluation
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_output_pred = model(test_input.to(device))
        mean_prediction = test_output_pred.mean.cpu().numpy()
        true_output = test_output.cpu().numpy()


    r2 = r2_score(true_output, mean_prediction)
    rmse = np.sqrt(mean_squared_error(true_output, mean_prediction))
    cd = chamfer_distance(torch.tensor(mean_prediction), torch.tensor(true_output)).item()

    print(f"\n\n\n--------------------------------------\n--------------------------------------\n--------------------------------------\n")
    print(f"IMAGE NAME: {image_name}")
    print(f'R^2 Score: {r2:.3f}, RMSE: {rmse:.3f}, Chamfer Distance: {cd:.3f}')


    # Per-task R² to catch any silent partial mismatch
    r2_all  = r2_score(true_output, mean_prediction)
    r2_task = r2_score(true_output, mean_prediction, multioutput='raw_values')
    print("R² overall:", r2_all, " | per-task:", r2_task)

    # Permutation sanity check: should degrade dramatically if pairing were random
    rng = np.random.default_rng(0)
    shuf = true_output.copy()
    rng.shuffle(shuf, axis=0)
    r2_shuf = r2_score(shuf, mean_prediction)
    print("R² with shuffled truths (should be << 0):", r2_shuf)
