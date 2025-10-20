import os
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from PIL import Image

# ==============================
# configurations
# ==============================
SHIFT_BOX_FLAG = -1
PLOT_SAVE_FOLDER = '/home/icclab/Documents/yyl/RadarDIP/RadarINV/results_radial_allplots'
os.makedirs(PLOT_SAVE_FOLDER, exist_ok=True)

data_root = '/home/icclab/Documents/yyl/RADIal_Dataset/RADIal/dataset_with_labels'
DATA_PATHS = {
    "image": os.path.join(data_root, 'images'),
    "lidar": os.path.join(data_root, 'lidars_no_ground'),
    "radar_ra": os.path.join(data_root, 'radars_ra'),
    "radar_pcd": os.path.join(data_root, 'radars_pcd'),
    "radar_psld": '/home/icclab/Documents/yyl/RadarDIP/RadarINV/results_radial_radarINV',
    # "radar_unet": "...",
    # "radar_latent_diffusion": "...",
    # "radar_l1": "...",
    # "radar_l2": "...",
}

INFO_FILE = '/home/icclab/Documents/yyl/RADIal_Dataset/RADIal/dataset_with_labels/radialx_infos_trainval.pkl'


# ==============================
# tool functions
# ==============================
def polar_to_cartesian(polar_image):
    """polar to cartesian conversion for radar RA heatmap"""
    height, width = polar_image.shape
    cartesian_image = np.zeros((height, 768), dtype=polar_image.dtype)

    range_reso = 103 / 512
    angle_reso = 180 / 751

    new_x_coord = np.linspace(-30, 30, 768)
    new_y_coord = np.linspace(0, 80, 512)
    center_x = new_x_coord[len(new_x_coord) // 2]

    for x in range(768):
        for y in range(512):
            dx = new_x_coord[x] - center_x + 1e-6
            dy = new_y_coord[y] + 1e-6
            r = np.sqrt(dx ** 2 + dy ** 2)
            theta = np.degrees(np.arctan(dx / dy)) + 90

            r_idx = int(np.round(r / range_reso))
            theta_idx = int(np.round(theta / angle_reso))

            if 0 <= r_idx < 512 and 0 <= theta_idx < 751:
                cartesian_image[y, x] = polar_image[r_idx, theta_idx]

    return cartesian_image, new_x_coord, new_y_coord


def filter_outliers(points, k=10, std_dev_threshold=2.0):
    """knn filter to remove outliers from point cloud"""
    if len(points) < k + 1:
        return points

    tree = KDTree(points)
    distances, _ = tree.query(points, k=k + 1)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    std_dev = np.std(mean_distances)
    mask = mean_distances < (np.mean(mean_distances) + std_dev_threshold * std_dev)
    return points[mask]


def load_bin_points(folder, fname, dims):
    """load point cloud from .bin file"""
    path = os.path.join(folder, f"{fname}.bin")
    return np.fromfile(path, dtype=np.float32).reshape(-1, dims)


def load_and_preprocess_frame(fname):
    """load and preprocess all data for a given frame"""
    data = {}

    # Image
    img_path = os.path.join(DATA_PATHS['image'], f"{fname}.png")
    if os.path.exists(img_path):
        data['image'] = np.array(Image.open(img_path))
    else:
        data['image'] = np.zeros((512, 512, 3), dtype=np.uint8)

    # Lidar
    data['lidar'] = load_bin_points(DATA_PATHS['lidar'], fname, 3)

    # RA heatmap
    ra = np.fromfile(os.path.join(DATA_PATHS['radar_ra'], f"{fname}.bin"), dtype=np.float64).reshape(512, 751)
    data['ra_heatmap'], _, _ = polar_to_cartesian(ra)

    # Radar cfar PCD
    data['radar_pcd'] = load_bin_points(DATA_PATHS['radar_pcd'], fname, 4)

    # Radar ours pcd
    psld_fname = os.path.join(DATA_PATHS['radar_psld'], fname, 'radar_points', fname+'.bin')
    data['radar_psld'] = np.fromfile(psld_fname, dtype=np.float32).reshape(-1, 3)

    # Other algorithms (if available)
    for key in DATA_PATHS.keys():
        if key in ['image', 'lidar', 'radar_pcd', 'radar_ra', 'radar_psld']:
            continue
        try:
            pts = load_bin_points(DATA_PATHS[key], fname, 3)
            data[key] = pts
        except FileNotFoundError:
            data[key] = np.zeros((0, 3))

    return data


def plot_pointclouds(data, fname):
    num_data = len(data.keys())
    fig, axs = plt.subplots(1, num_data, figsize=(int(5 * num_data), 5))
    plt.style.use('default')

    if num_data == 1:
        axs = [axs]

    scatter_cfg_gt = dict(color='blue', s=0.1, marker='o')
    scatter_cfg_other = dict(color='red', s=0.2, marker='o')
    axis_cfg = dict(xlim=(-30, 30), ylim=(0, 100))

    idx = 0

    # Image
    if 'image' in data:
        axs[idx].imshow(data['image'])
        axs[idx].axis('off')
        idx += 1

    # RA heatmap
    if 'ra_heatmap' in data:
        axs[idx].imshow(data['ra_heatmap'])
        axs[idx].invert_yaxis()
        axs[idx].axis('off')
        idx += 1

    # Radar PCD + lidar
    if 'radar_pcd' in data and data['radar_pcd'].size > 0:
        axs[idx].scatter(data['radar_pcd'][:, 1], data['radar_pcd'][:, 0], **scatter_cfg_other)
        axs[idx].scatter(data['lidar'][:, 1], data['lidar'][:, 0], **scatter_cfg_gt)
        axs[idx].set(**axis_cfg)
        axs[idx].axis('off')
        idx += 1

    # Other algorithms
    for key in data:
        if key in ['image', 'ra_heatmap', 'radar_ra', 'radar_pcd', 'lidar']:
            continue
        if data[key].size == 0:
            continue
        axs[idx].scatter(data[key][:, 1], data[key][:, 0], **scatter_cfg_other)
        axs[idx].scatter(data['lidar'][:, 1], data['lidar'][:, 0], **scatter_cfg_gt)
        axs[idx].set(**axis_cfg)
        axs[idx].axis('off')
        idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_FOLDER, f"{fname}.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)


def main():
    with open(INFO_FILE, 'rb') as f:
        infos = pickle.load(f)

    fnames = [
        Path(info['radar']['radars_ra_path']).stem
        for info in infos if 'radar' in info and 'radars_ra_path' in info['radar']
    ]

    for fname in fnames:
        print(f"Processing frame: {fname}")
        try:
            frame_data = load_and_preprocess_frame(fname)
            plot_pointclouds(frame_data, fname)
        except Exception as e:
            print(f"❌ Error in frame {fname}: {e}")
            continue


if __name__ == "__main__":
    main()
