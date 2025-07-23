import numpy as np
import matplotlib.pyplot as plt
import os
from config import POINTS3D_PATH, IMAGES_TXT_PATH, DEPTH_FILE_PATH, BASE_DIR, SCENE_NAME, TEST_VAR, PREDICT_MEAN

# Load the predicted variance data
predicted_var = np.load(TEST_VAR)[0]
predicted_variance = np.array(predicted_var).reshape(-1, 6)

# Create var_figure folder in the scene directory if it doesn't exist
scene_dir = os.path.dirname(TEST_VAR)
var_figure_dir = os.path.join(scene_dir, "var_figure")
os.makedirs(var_figure_dir, exist_ok=True)

# Filtering based on mean RGB variance
rgb_var = predicted_variance[:, 3:6]
mean_rgb_var = np.mean(rgb_var, axis=1)
threshold = np.percentile(mean_rgb_var, 50)
filtered_indices = mean_rgb_var <= threshold
filtered_variance = predicted_variance[filtered_indices]

#debuging 
print("Total points:", len(predicted_variance))
print("Points after filtering:", np.sum(filtered_indices))
print("Points removed:", len(predicted_variance) - np.sum(filtered_indices))

# Compute y-limits from unfiltered data
spatial_ylim = [np.min(predicted_variance[:, :3]), np.max(predicted_variance[:, :3])]
color_ylim = [np.min(predicted_variance[:, 3:]), np.max(predicted_variance[:, 3:])]
ylims = [spatial_ylim, color_ylim]

def plot_variances(variance, title_prefix, save_path, ylims=None):
    colors = ['#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4', '#46F0F0']
    labels = ['Variance in x', 'Variance in y', 'Variance in z', 'Variance in r', 'Variance in g', 'Variance in b']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Top: spatial variances
    for i in range(3):
        axes[0].plot(range(len(variance)), variance[:, i], label=labels[i], color=colors[i], linewidth=1)
    axes[0].legend(fontsize=12, loc='upper right', frameon=False)
    axes[0].set_ylabel('Spatial Variance')
    if ylims is not None:
        axes[0].set_ylim(ylims[0])

    # Bottom: color variances
    for i in range(3, 6):
        axes[1].plot(range(len(variance)), variance[:, i], label=labels[i], color=colors[i], linewidth=1)
    axes[1].legend(fontsize=12, loc='upper right', frameon=False)
    axes[1].set_ylabel('Color Variance')
    axes[1].set_xlabel('Sample Index')
    if ylims is not None:
        axes[1].set_ylim(ylims[1])

    fig.suptitle(f'{title_prefix} Variance', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Plot unfiltered
plot_variances(predicted_variance, "Unfiltered", os.path.join(var_figure_dir, "variance_plots_unfiltered.png"))
# Plot filtered
plot_variances(filtered_variance, "Filtered", os.path.join(var_figure_dir, "variance_plots_filtered.png"))



plt.hist(mean_rgb_var, bins=50)
plt.title("Histogram of Mean RGB Variance")
plt.xlabel("Mean RGB Variance")
plt.ylabel("Count")
plt.show()