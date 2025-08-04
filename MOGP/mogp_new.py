import torch
import time
import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import gpytorch

# Import your modular components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MOGP.gp_models import (
    LCMMultiTaskGPModel, 
    IndependentMultiTaskGPModel, 
    LCMMixedMaternModel,
    LCMMaternWendlandModel
)
from MOGP.experiment_configs import EXPERIMENT_CONFIGS
from config import POINTS3D_PATH, DEPTH_FILE_PATH, IMAGES_TXT_PATH, BASE_DIR, SCENE_NAME

# Constants
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

RADIUS = 0.25
DYNAMIC_MVNTS = 8
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_EPOCHS = 10
NUM_TASKS = 6
RANK = 4
NUM_INDUCING = 200
LR = 0.2
WEIGHT_DECAY = 1e-6 
NU = 0.5
images = ["000115.JPG", "000117.JPG", "000079.JPG", "000072.JPG"]

# Set up output directory
GP_OUTPUT_DIR = os.path.join(BASE_DIR, "gp")
os.makedirs(GP_OUTPUT_DIR, exist_ok=True)

# Data loading functions (same as your current file)
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
    dir_depth = os.path.dirname(depth_file_path)
    order_file = os.path.join(dir_depth, "image_list_colmap.txt")
    if os.path.exists(order_file):
        with open(order_file) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return load_image_order_from_txt(fallback_images_txt)

def collapse_depth_2d(D):
    if D.ndim == 2:
        return D.astype(np.float32)
    if D.shape[2] >= 3:
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return (D[..., :3].astype(np.float32) @ w).astype(np.float32)
    return D[..., 0].astype(np.float32)

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
                       radius_factor=RADIUS, num_samples=DYNAMIC_MVNTS):
    depth_images = np.load(depth_file_path)
    print("depth stack shape:", depth_images.shape)
    
    ordered_names = load_stack_order(depth_file_path, IMAGES_TXT_PATH)
    image_indices = {name: i for i, name in enumerate(ordered_names)}
    
    data_by_image = {}
    
    for image_name, data_points in valid_data.items():
        if image_name not in image_indices:
            continue
        
        D = depth_images[image_indices[image_name]]
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
        r = int(radius_factor * min(H, W))
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        movements = [(int(r*np.cos(a)), int(r*np.sin(a))) for a in angles]
        
        input_data, output_data, test_data = [], [], []
        for p in data_points:
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < W and 0 <= y < H):
                continue
            d = float(D[y, x])
            nd = (d - min_depth) / (max_depth - min_depth + 1e-12)
            input_data.append([x / W, y / H, nd])
            output_data.append(p[2:])
            
            for dx, dy in movements:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    nd2 = (float(D[ny, nx]) - min_depth) / (max_depth - min_depth + 1e-12)
                    test_data.append([nx / W, ny / H, nd2])
        
        data_by_image[image_name] = {
            'input': np.asarray(input_data, dtype=float),
            'output': np.asarray(output_data, dtype=float),
            'test': np.asarray(test_data, dtype=float),
        }
    return data_by_image

def chamfer_distance(pred_points, true_points):
    pred_expanded = pred_points.unsqueeze(1)
    true_expanded = true_points.unsqueeze(0)
    distances = torch.norm(pred_expanded - true_expanded, dim=-1)
    forward_cd = torch.mean(torch.min(distances, dim=1)[0])
    backward_cd = torch.mean(torch.min(distances, dim=0)[0])
    return forward_cd + backward_cd

def save_model_files(experiment_name, image_name, model_info, base_dir=GP_OUTPUT_DIR, scene_name=SCENE_NAME):
    """Save model files with experiment and image names"""
    base_name = f"{scene_name}_{experiment_name}_{image_name}"
    
    model_save_path = os.path.join(base_dir, f"{base_name}.pth")
    likelihood_save_path = os.path.join(base_dir, f"{base_name}likelihood.pth")
    losses_save_path = os.path.join(base_dir, f"{base_name}.npy")
    mean_save_path = os.path.join(base_dir, f"{base_name}mean.npy")
    variance_save_path = os.path.join(base_dir, f"{base_name}test_var.npy")
    test_input_save_path = os.path.join(base_dir, f"{base_name}test_input.npy")
    test_output_save_path = os.path.join(base_dir, f"{base_name}test_output.npy")
    
    torch.save(model_info['model'].state_dict(), model_save_path)
    torch.save(model_info['likelihood'].state_dict(), likelihood_save_path)
    np.save(losses_save_path, np.array(model_info['losses']))
    np.save(mean_save_path, model_info['test_mean'])
    np.save(variance_save_path, model_info['test_variance'])
    np.save(test_input_save_path, model_info['test_input'])
    np.save(test_output_save_path, model_info['test_output'])
    
    print(f"  Model files saved:")
    print(f"  Model: {model_save_path}")
    print(f"  Likelihood: {likelihood_save_path}")
    print(f"  Losses: {losses_save_path}")
    print(f"  Mean predictions: {mean_save_path}")
    print(f"  Variance predictions: {variance_save_path}")
    print(f"  Test inputs: {test_input_save_path}")
    print(f"  Test outputs: {test_output_save_path}")

#Run a single experiment with given configuration"""
def run_experiment(experiment_name, model_class_name, model_params, images, valid_data, depth_file_path, min_depth, max_depth):
   
    
    # Get model class
    model_classes = {
    "LCMMultiTaskGPModel": LCMMultiTaskGPModel,
    "IndependentMultiTaskGPModel": IndependentMultiTaskGPModel,
    "LCMMixedMaternModel": LCMMixedMaternModel,
    "LCMMaternWendlandModel": LCMMaternWendlandModel
}

    ModelClass = model_classes[model_class_name]
    
    results = []
    models_info = {}
    
    for image_name in images:
        print(f"Processing {experiment_name} - {image_name}")
        
        # Filter data for this specific image
        valid_data_single = {k: v for k, v in valid_data.items() if k == image_name}
        data_by_image_single = generate_test_data(valid_data_single, depth_file_path, min_depth, max_depth)
        
        if image_name not in data_by_image_single:
            print(f"Warning: No data found for image {image_name}, skipping...")
            continue
        
        # Get and prepare data
        input_data = data_by_image_single[image_name]['input']
        output_data = data_by_image_single[image_name]['output']
        
        input_data_normalized = input_data
        scaler_output = MinMaxScaler().fit(output_data)
        output_data_normalized = scaler_output.transform(output_data)
        
        train_input, test_input, train_output, test_output = train_test_split(
            input_data_normalized, output_data_normalized, 
            test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        train_input = torch.tensor(train_input, dtype=DTYPE)
        train_output = torch.tensor(train_output, dtype=DTYPE)
        test_input = torch.tensor(test_input, dtype=DTYPE)
        test_output = torch.tensor(test_output, dtype=DTYPE)
        
        print(f"Training data shape: {train_input.shape}")
        print(f"Test data shape: {test_input.shape}")
        
        # Training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=NUM_TASKS).to(device)
        
        # Create model with experiment parameters
        if model_class_name == "LCMMultiTaskGPModel":
            model = ModelClass(train_input.to(device), train_output.to(device), likelihood, 
                             num_tasks=NUM_TASKS, **model_params).to(device)
        elif model_class_name == "LCMMixedKernelModel":
            model = ModelClass(train_input.to(device), train_output.to(device), likelihood, 
                             num_tasks=NUM_TASKS, **model_params).to(device)
        else:
            model = ModelClass(train_input.to(device), train_output.to(device), likelihood, 
                             num_tasks=NUM_TASKS, **model_params).to(device)
        
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        # Training loop
        losses = []
        model.train()
        likelihood.train()
        
        train_start = time.time()
        for i in tqdm(range(NUM_EPOCHS), desc=f"Training {experiment_name} - {image_name}"):
            optimizer.zero_grad()
            output = model(train_input.to(device))
            loss = -mll(output, train_output.to(device))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        train_end = time.time()
        train_time = train_end - train_start
        
        # Evaluation
        model.eval()
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_output_pred = model(test_input.to(device))
            mean_prediction = test_output_pred.mean.cpu().numpy()
            true_output = test_output.cpu().numpy()
        
        # Calculate metrics
        r2_total = r2_score(true_output, mean_prediction)
        r2_per_task = r2_score(true_output, mean_prediction, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(true_output, mean_prediction))
        cd = chamfer_distance(torch.tensor(mean_prediction), torch.tensor(true_output)).item()
        
        # Store results
        result = {
            'experiment': experiment_name,
            'image_name': image_name,
            'r2_total': r2_total,
            'r2_per_task': r2_per_task,
            'rmse': rmse,
            'chamfer_distance': cd,
            'train_time': train_time,
            'train_samples': len(train_input),
            'test_samples': len(test_input)
        }
        results.append(result)
        
        # Store model info for saving
        models_info[image_name] = {
            'model': model,
            'likelihood': likelihood,
            'losses': losses,
            'scaler_output': scaler_output,
            'test_mean': mean_prediction,
            'test_variance': test_output_pred.variance.cpu().numpy(),
            'test_input': test_input.cpu().numpy(),
            'test_output': true_output
        }
        
        print(f"Results for {experiment_name} - {image_name}:")
        print(f"R² Total: {r2_total:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Chamfer Distance: {cd:.3f}")
        print(f"Training time: {train_time:.2f} seconds")
        print("-" * 50)
    
    return results, models_info

def main():
    """Main function to run experiments"""
    
    # Load data
    print("Loading data...")
    points3d_dict = load_points3D(POINTS3D_PATH)
    valid_data = parse_images_file(IMAGES_TXT_PATH, points3d_dict)
    
    depth_images = np.load(DEPTH_FILE_PATH)
    ordered_names = load_stack_order(DEPTH_FILE_PATH, IMAGES_TXT_PATH)
    image_indices = {name: i for i, name in enumerate(ordered_names)}
    
    all_depths = []
    for img_name, data_points in valid_data.items():
        D = depth_images[image_indices[img_name]]
        current_depth_image = collapse_depth_2d(D)
        img_height, img_width = current_depth_image.shape
        for point in data_points:
            x, y = int(point[0]), int(point[1])
            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                continue
            original_depth = current_depth_image[y, x]
            all_depths.append(original_depth)
    
    min_depth = np.min(all_depths)
    max_depth = np.max(all_depths)
    
    # Run experiments
    all_results = []
    all_models_info = {}
    
    for exp_name, exp_config in EXPERIMENT_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT: {exp_name}")
        print(f"Description: {exp_config['description']}")
        print(f"{'='*80}")
        
        results, models_info = run_experiment(
            exp_name, 
            exp_config['model_class'], 
            exp_config['params'],
            images, valid_data, DEPTH_FILE_PATH, min_depth, max_depth
        )
        
        all_results.extend(results)
        all_models_info[exp_name] = models_info
        
        # Save results for this experiment
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(GP_OUTPUT_DIR, f"{SCENE_NAME}_{exp_name}_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to: {results_csv_path}")
    
    # Create summary of all experiments
    summary_df = pd.DataFrame(all_results)
    summary_csv_path = os.path.join(GP_OUTPUT_DIR, f"{SCENE_NAME}_all_experiments_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"\nAll experiments summary saved to: {summary_csv_path}")
    print("\nExperiment Summary:")
    print(summary_df.groupby('experiment')['r2_total'].agg(['mean', 'std']))
    
    # Model saving section
    print("\n" + "="*80)
    print("SAVE TRAINING INFORMATION")
    print("="*80)
    
    # Show performance summary first
    print("EXPERIMENT PERFORMANCE SUMMARY:")
    print("-" * 50)
    for exp in EXPERIMENT_CONFIGS.keys():
        exp_results = summary_df[summary_df['experiment'] == exp]
        if not exp_results.empty:
            print(f"{exp}:")
            print(f"  Avg R²: {exp_results['r2_total'].mean():.3f} ± {exp_results['r2_total'].std():.3f}")
            print(f"  Avg RMSE: {exp_results['rmse'].mean():.3f} ± {exp_results['rmse'].std():.3f}")
            best_img = exp_results.loc[exp_results['r2_total'].idxmax(), 'image_name']
            print(f"  Best image: {best_img}")
            print()
    
    # Best combination suggestion
    best_result = summary_df.loc[summary_df['r2_total'].idxmax()]
    best_exp = best_result['experiment']
    best_img = best_result['image_name']
    
    print("    RECOMMENDED (Best Overall Performance):")
    print(f"   Experiment: {best_exp}")
    print(f"   Image: {best_img}")
    print(f"   R² Score: {best_result['r2_total']:.3f}")
    print(f"   RMSE: {best_result['rmse']:.3f}")
    print(f"   Chamfer Distance: {best_result['chamfer_distance']:.3f}")
    
    best_choice = input(f"\nSave this best model? (y/n): ")
    saved_models = []
    
    if best_choice.lower() == 'y':
        print(f"\nSaving best model ({best_exp} + {best_img})...")
        save_model_files(best_exp, best_img, all_models_info[best_exp][best_img])
        saved_models.append((best_exp, best_img))
        print("Best model saved!")
    
    # Manual selection
    print(f"\n" + "="*50)
    print("OR CHOOSE CUSTOM COMBINATION:")
    print("="*50)
    
    # Step 1: Choose experiment
    print("Available experiments:")
    for i, exp_name in enumerate(EXPERIMENT_CONFIGS.keys()):
        exp_results = summary_df[summary_df['experiment'] == exp_name]
        if not exp_results.empty:
            avg_r2 = exp_results['r2_total'].mean()
            best_r2 = exp_results['r2_total'].max()
            print(f"{i+1}. {exp_name}")
            print(f"   Avg R²: {avg_r2:.3f}, Best R²: {best_r2:.3f}")
    
    exp_choice = input(f"\nEnter experiment number (1-{len(EXPERIMENT_CONFIGS)}) or 'skip': ")
    
    if exp_choice.lower() != 'skip':
        exp_idx = int(exp_choice) - 1
        selected_exp = list(EXPERIMENT_CONFIGS.keys())[exp_idx]
        
        # Step 2: Choose image
        print(f"\nImages for {selected_exp}:")
        exp_images = summary_df[summary_df['experiment'] == selected_exp]['image_name'].unique()
        
        for i, img in enumerate(exp_images):
            img_result = summary_df[(summary_df['experiment'] == selected_exp) & 
                                   (summary_df['image_name'] == img)].iloc[0]
            print(f"{i+1}. {img}")
            print(f"   R²: {img_result['r2_total']:.3f}, RMSE: {img_result['rmse']:.3f}")
        
        img_choice = input(f"\nEnter image number (1-{len(exp_images)}) or 'skip': ")
        
        if img_choice.lower() != 'skip':
            img_idx = int(img_choice) - 1
            selected_image = exp_images[img_idx]
            
            # Check if already saved
            if (selected_exp, selected_image) not in saved_models:
                print(f"\n Saving custom model ({selected_exp} + {selected_image})...")
                save_model_files(selected_exp, selected_image, all_models_info[selected_exp][selected_image])
                saved_models.append((selected_exp, selected_image))
                print("Custom model saved!")
            else:
                print(f"Model for {selected_exp} + {selected_image} already saved!")
    
    # Final summary
    print(f"\n" + "="*80)
    print("FINAL SUMMARY:")
    print("="*80)
    if saved_models:
        print("Saved models:")
        for exp, img in saved_models:
            result = summary_df[(summary_df['experiment'] == exp) & 
                               (summary_df['image_name'] == img)].iloc[0]
            print(f"  • {exp} + {img} (R²: {result['r2_total']:.3f})")
    else:
        print("No models were saved.")

if __name__ == "__main__":
    main()