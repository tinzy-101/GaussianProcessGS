import torch
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

from config import (
    POINTS3D_PATH, 
    DEPTH_FILE_PATH, 
    IMAGES_TXT_PATH,
    BASE_DIR,
    SCENE_NAME
)
TEST_SIZE = 0.2
RANDOM_STATE = 42
TRAINING_ITERATIONS = 300
NUM_TASKS = 6
LEARNING_RATE = 0.1
WEIGHT_DECAY =1e-6 
NU = 0.5

#Set up output directory if it doesn't exist
GP_OUTPUT_DIR = os.path.join(BASE_DIR, "gp")
os.makedirs(GP_OUTPUT_DIR, exist_ok= True)


# Prepare data paths
file_path_points3d = POINTS3D_PATH
depth_file_path = DEPTH_FILE_PATH
file_path_images = IMAGES_TXT_PATH

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

def generate_training_data(valid_data, depth_file_path,min_depth, max_depth):
    """
    Generate training data from pixel-to-point correspondences only.
    """
    depth_images = np.load(depth_file_path)
    image_indices = {name: idx for idx, name in enumerate(sorted(valid_data.keys()))}

    data_by_image = {}

    for image_name, data_points in valid_data.items():
        current_depth_image = depth_images[image_indices[image_name]]
        image_height, image_width = current_depth_image.shape

        input_data = []
        output_data = []

        for point in data_points:
            x, y = int(point[0]), int(point[1])
            if x < 0 or x >= image_width or y < 0 or y >= image_height:
                continue
            original_depth = current_depth_image[y, x]
            normalized_depth = (original_depth - min_depth) / (max_depth - min_depth)
            input_data.append([x, y,normalized_depth])
            output_data.append(point[2:])

        input_data = np.array(input_data, dtype=float)
        input_data[:, 0] /= image_width  # Normalize x to [0, 1]
        input_data[:, 1] /= image_height  # Normalize y to [0, 1]
        data_by_image[image_name] = {
            'input': input_data,
            'output': np.array(output_data, dtype=float)
        }

    return data_by_image


#Load and Preprocess data
points3d_dict = load_points3D(file_path_points3d)
valid_data = parse_images_file(file_path_images, points3d_dict)

#Load Depth images and normalize
depth_images = np.load(depth_file_path)
image_indices = {name: i for i, name in enumerate(sorted(valid_data.keys()))}

all_depths =[]

for img_name, data_points in valid_data.items():
    current_depth_image = depth_images[image_indices[img_name]]
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
top_images_path = os.path.join(BASE_DIR,"top_four_images.json")
with open(top_images_path, "r") as f:
    top_image_names = json.load(f)

#filter on the top 4 images
valid_data = {k: v for k, v in valid_data.items() if k in top_image_names}

#Generate input/ouput/test sets
data_by_image_new = generate_training_data(valid_data, depth_file_path, min_depth,max_depth)



# Stack input and output data from the selected images
all_input_data = []
all_output_data = []

for img_name in top_image_names:
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
train_input = torch.tensor(train_input, dtype=torch.float32)
train_output = torch.tensor(train_output, dtype=torch.float32)
test_input = torch.tensor(test_input, dtype=torch.float32)
test_output = torch.tensor(test_output, dtype=torch.float32)

import pandas as pd
train_x_df = pd.DataFrame(train_input)
print(train_x_df.describe())
train_y_df = pd.DataFrame(train_output)
print(train_y_df.describe())

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
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

losses = []
model.train()
likelihood.train()
for i in range(TRAINING_ITERATIONS):
    optimizer.zero_grad()
    output = model(train_input.to(device))
    loss = -mll(output, train_output.to(device))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f'Iteration {i + 1}/{TRAINING_ITERATIONS}, Loss: {loss.item()}')


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

print(f'R^2 Score: {r2:.3f}, RMSE: {rmse:.3f}, Chamfer Distance: {cd:.3f}')

