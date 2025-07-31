import open3d as o3d
import numpy as np
import argparse
import os

def read_points3D_text_robust(path):
    """Robust version that handles both integer and float RGB values"""
    xyzs, rgbs, errors = [], [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            elems = line.split()
            xyzs.append([float(e) for e in elems[1:4]])
            
            # Handle RGB values - try integer first, then float
            try:
                rgb = [int(e) for e in elems[4:7]]
            except ValueError:
                # If integers fail, try floats and convert to 0-255 range
                rgb = [int(float(e) * 255) for e in elems[4:7]]
                # Clamp to valid range
                rgb = [max(0, min(255, r)) for r in rgb]
            
            rgbs.append(rgb)
            errors.append(float(elems[7]))
    
    return np.array(xyzs), np.array(rgbs), np.array(errors)

def load_point_cloud_from_colmap(file_path):
    """Load point cloud from COLMAP format and convert to Open3D format"""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return None
    
    try:
        # Read points using robust function
        xyzs, rgbs, errors = read_points3D_text_robust(file_path)
        print(f"Loaded {len(xyzs)} points from {file_path}")
        print(f"XYZ range: X[{xyzs[:, 0].min():.3f}, {xyzs[:, 0].max():.3f}], "
              f"Y[{xyzs[:, 1].min():.3f}, {xyzs[:, 1].max():.3f}], "
              f"Z[{xyzs[:, 2].min():.3f}, {xyzs[:, 2].max():.3f}]")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        pcd.colors = o3d.utility.Vector3dVector(rgbs / 255.0)  # Normalize colors to [0,1]
        
        return pcd
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def visualize_point_clouds_comparison(scene_name, base_path="mipnerf360"):
    """Visualize sparse vs densified point clouds side by side"""
    
    # Construct file paths
    sparse_path = os.path.join(base_path, scene_name, "sparse", "0", "points3D.txt")
    densified_path = os.path.join(base_path, scene_name, f"pixel-to-point-{scene_name}", "points3D.txt")
    
    print(f"Checking file paths:")
    print(f"Sparse: {sparse_path} - Exists: {os.path.exists(sparse_path)}")
    print(f"Densified: {densified_path} - Exists: {os.path.exists(densified_path)}")
    
    print(f"\nLoading sparse point cloud from: {sparse_path}")
    sparse_pcd = load_point_cloud_from_colmap(sparse_path)
    
    print(f"\nLoading densified point cloud from: {densified_path}")
    densified_pcd = load_point_cloud_from_colmap(densified_path)
    
    if sparse_pcd is None and densified_pcd is None:
        print("Error: Could not load any point clouds")
        return
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"Point Cloud Comparison - {scene_name}", width=1600, height=800)
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coordinate_frame)
    
    # Create a combined point cloud with different colors
    combined_pcd = o3d.geometry.PointCloud()
    
    if sparse_pcd is not None and densified_pcd is not None:
        # Combine points from both clouds
        sparse_points = np.asarray(sparse_pcd.points)
        densified_points = np.asarray(densified_pcd.points)
        
        # Create colors for sparse points (blue)
        sparse_colors = np.ones((len(sparse_points), 3)) * [0, 0, 1]  # Blue
        
        # Create colors for densified points (red)
        densified_colors = np.ones((len(densified_points), 3)) * [1, 0, 0]  # Red
        
        # Combine points and colors
        all_points = np.vstack([sparse_points, densified_points])
        all_colors = np.vstack([sparse_colors, densified_colors])
        
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        print(f"Combined visualization:")
        print(f"  Blue points: {len(sparse_points)} sparse points")
        print(f"  Red points: {len(densified_points)} densified points")
        print(f"  Total points: {len(all_points)}")
        
    elif sparse_pcd is not None:
        # Only sparse available
        combined_pcd = sparse_pcd
        combined_pcd.paint_uniform_color([0, 0, 1])  # Blue
        print(f"Sparse point cloud only: {len(sparse_pcd.points)} points (BLUE)")
        
    elif densified_pcd is not None:
        # Only densified available
        combined_pcd = densified_pcd
        combined_pcd.paint_uniform_color([1, 0, 0])  # Red
        print(f"Densified point cloud only: {len(densified_pcd.points)} points (RED)")
    
    # Add the combined point cloud to visualizer
    vis.add_geometry(combined_pcd)
    
    # Set up view
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = [0.1, 0.1, 0.1]  # Dark background
    
    # Add legend
    print("\nLegend:")
    if sparse_pcd is not None:
        print("Blue points: Sparse point cloud (COLMAP)")
    if densified_pcd is not None:
        print("Red points: Densified point cloud (Pixel-to-Point)")
    print("\nControls:")
    print("- Mouse: Rotate view")
    print("- Mouse wheel: Zoom")
    print("- Shift + Mouse: Pan")
    print("- R: Reset view")
    print("- Q: Quit")
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Visualize sparse vs densified point clouds")
    parser.add_argument("--scene", type=str, default="flowers", help="Scene name to visualize")
    parser.add_argument("--base-path", type=str, default="mipnerf360", help="Base path to dataset")
    
    args = parser.parse_args()
    
    visualize_point_clouds_comparison(args.scene, args.base_path)

if __name__ == "__main__":
    main()