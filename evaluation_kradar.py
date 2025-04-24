# 画在kradar数据集下的对比结果
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import cv2
from pytorch_fid import fid_score
from torchvision.models import inception_v3

inception_v3 = inception_v3(pretrained=True)

fname_list = []
valid_fname_list = [
    'radar_1-cube_00037-lidar_1-os2-64_00006',
    'radar_1-cube_00050-lidar_1-os2-64_00019',
    'radar_1-cube_00051-lidar_1-os2-64_00020',
    'radar_1-cube_00097-lidar_1-os2-64_00066',
    'radar_1-cube_00037-lidar_1-os2-64_00006',
    'radar_1-cube_00132-lidar_1-os2-64_00101'
]

def calculate_metrics(x, y):
    """
    Calculate various metrics between two point clouds: 
    Fréchet Inception Distance (FID), Chamfer Distance (CD), 
    Uniform Chamfer Distance (UCD), Modified Hausdorff Distance (MHD), 
    and Uniform Modified Hausdorff Distance (UMHD).

    Parameters:
    x (np.ndarray): First point cloud of shape (N, D) where N is the number of points and D is the dimensionality.
    y (np.ndarray): Second point cloud of shape (M, D) where M is the number of points and D is the dimensionality.

    Returns:
    dict: A dictionary containing the calculated metrics.
    """
    if x.shape[0] != y.shape[0]:
        min_points = min(x.shape[0], y.shape[0]) // 2
        x_indices = np.random.choice(x.shape[0], min_points, replace=False)
        y_indices = np.random.choice(y.shape[0], min_points, replace=False)
        x = x[x_indices]
        y = y[y_indices]
    # Chamfer Distance
    x = np.expand_dims(x, axis=1)  # Shape (N, 1, D)
    y = np.expand_dims(y, axis=0)  # Shape (1, M, D)
    dist = np.sum((x - y) ** 2, axis=-1)  # Shape (N, M)
    min_dist_x_to_y = np.min(dist, axis=1)  # Shape (N,)
    min_dist_y_to_x = np.min(dist, axis=0)  # Shape (M,)
    chamfer_dist = np.mean(min_dist_x_to_y) + np.mean(min_dist_y_to_x)  # Scalar

    # Uniform Chamfer Distance (UCD)
    ucd = np.mean(min_dist_x_to_y)  # Single direction mean distance

    # Modified Hausdorff Distance (MHD)
    mhd = max(np.mean(min_dist_x_to_y), np.mean(min_dist_y_to_x))

    # Uniform Modified Hausdorff Distance (UMHD)
    umhd = np.max(min_dist_x_to_y)  # Single direction max distance

    # fid = fid_score.FID(x, y, inception_v3)

    return {
        # 'fid': fid,
        'cd': chamfer_dist,
        'ucd': ucd,
        'mhd': mhd,
        'umhd': umhd
    }

def polar_to_cartesian(out_ra_np, range_coord=None, azimuth_coord=None, x_coord=None, y_coord=None):
    if range_coord is None:
        range_coord = np.linspace(0,out_ra_np.shape[-2],out_ra_np.shape[-2])/out_ra_np.shape[-2]*118.037
    if azimuth_coord is None:
        azimuth_coord = np.linspace(-53, 53, out_ra_np.shape[-1])

    if x_coord is None:
        x_coord = np.linspace(-50, 50, out_ra_np.shape[-1])
    if y_coord is None:
        y_coord = np.linspace(0,out_ra_np.shape[-2],out_ra_np.shape[-2])/out_ra_np.shape[-2]*118.037
    RA_cateresian = np.zeros((len(y_coord), len(x_coord)))
    range_resolution = range_coord[1]-range_coord[0]
    azimuth_resolution = azimuth_coord[1]-azimuth_coord[0]
    for i in range(len(x_coord)):
        for j in range(len(y_coord)):
            x = x_coord[i]
            y = y_coord[j]
            r = int(np.sqrt(x**2+y**2)/range_resolution) - 0
            a = int((np.arctan2(x, y)*180/np.pi + 53)/azimuth_resolution)
            if r >= out_ra_np.shape[-2] or a >= out_ra_np.shape[-1] or r < 0 or a < 0:
                continue
            RA_cateresian[j, i] = out_ra_np[r, a]
    return RA_cateresian

def numpy_cfar(radar, guard_cells=6, training_cells=3, false_alarm_rate=1):
    # Assuming radar is a 2D numpy array
    # radar = np.pad(radar, ((guard_cells, guard_cells), (guard_cells, guard_cells)), mode='constant', constant_values=0)
    # Calculate CFAR for x-axis and store threshold values
    x_cfar = np.zeros_like(radar)
    x_thresh = np.zeros_like(radar)
    for i in range(guard_cells, radar.shape[0] - guard_cells):
        for j in range(guard_cells, radar.shape[1] - guard_cells):
            training_window = radar[i-training_cells:i+training_cells+1, j-training_cells:j+training_cells+1]
            # Correctly remove the center cell from the training window
            training_window = np.delete(training_window, training_cells, axis=0)
            training_window = np.delete(training_window, training_cells, axis=1)
            noise_level = np.mean(training_window)
            threshold = noise_level * false_alarm_rate
            x_thresh[i, j] = threshold  # Store threshold value
            if radar[i, j] > threshold:
                x_cfar[i, j] = radar[i, j]
    
    # Calculate CFAR for y-axis and store threshold values
    y_cfar = np.zeros_like(radar)
    y_thresh = np.zeros_like(radar)
    for i in range(guard_cells, radar.shape[0] - guard_cells):
        for j in range(guard_cells, radar.shape[1] - guard_cells):
            training_window = radar[i-training_cells:i+training_cells+1, j-training_cells:j+training_cells+1]
            # Correctly remove the center cell from the training window
            training_window = np.delete(training_window, training_cells, axis=0)
            training_window = np.delete(training_window, training_cells, axis=1)
            noise_level = np.mean(training_window)
            threshold = noise_level * false_alarm_rate
            y_thresh[i, j] = threshold  # Store threshold value
            if radar[i, j] > threshold:
                y_cfar[i, j] = radar[i, j]
    
    # Combine x and y CFAR results
    combined_cfar = np.maximum(x_cfar, y_cfar)
    
    # Remove edge peaks with adjustable edge width
    edge_peaks_removed = np.zeros_like(combined_cfar)
    w_edge_width =26  # Adjustable edge width
    h_edge_width =80  # Adjustable edge width
    edge_peaks_removed[0:-h_edge_width, w_edge_width:-w_edge_width] = combined_cfar[0:-h_edge_width, w_edge_width:-w_edge_width]
    
    return edge_peaks_removed, np.logical_and(x_thresh, y_thresh)


def save_points_kradar(sample, fname, folder_path, is_lidar=False):

    sample = sample.squeeze()
    azimuth_coord = np.linspace(-53, 53, sample.shape[-1])
    # 预测
    out_ra = torch.sigmoid(1000*(sample-0.01))
    if len(out_ra.shape) >= 3:
        out_ra = out_ra.mean(dim=0)
    if out_ra.device.type == 'cuda':
        out_ra = out_ra.detach().cpu().numpy()
    out_ra_np = out_ra.squeeze()
    # threshold = 0.01
    threshold = out_ra_np.mean()+0.01
    pixel_indices = np.where(out_ra_np > threshold)
    real_coords = []
    range_coord = np.linspace(0,out_ra_np.shape[0],out_ra_np.shape[0])/out_ra_np.shape[0]*118.037
    for i in range(len(pixel_indices[0])):
        pixel_y, pixel_x = pixel_indices[0][i], pixel_indices[1][i]  # y corresponds to range, x to azimuth
        if pixel_x >= len(azimuth_coord) or pixel_y >= len(range_coord):
            continue
        azimuth = np.deg2rad(azimuth_coord[pixel_x])  # Convert pixel x to azimuth in radians
        # range_val = range_coord[out_ra_np.shape[0]-1-pixel_y]  # Convert pixel y to range，because coloradar is top-down
        range_val = range_coord[pixel_y]  # Convert pixel y to range
        x = range_val * np.cos(azimuth)  # Convert polar to Cartesian coordinates
        y = range_val * np.sin(azimuth)  # Convert polar to Cartesian coordinates
        real_coords.append([x, y, 0])  # Assuming 0 for the z-coordinate
    real_coords = np.array(real_coords).astype(np.float32)

    return real_coords


radar_heatmap_path = '/scratch/project/cpautodriving/yanlongyang/kradar/data'
L12_points_path = '/scratch/project/cpautodriving/yanlongyang/dataset_with_labels/results_kradar/'

lidar_points_list = []
radar_heatmap_list = []
for fname in valid_fname_list:
    lidar_points_file = os.path.join(L12_points_path, 'L2_Reg/lidar_points', f'{fname}.bin')
    if os.path.exists(lidar_points_file):
        lidar_points = np.fromfile(lidar_points_file, dtype=np.float32).reshape(-1, 3)
        lidar_points_list.append(lidar_points)
        print(fname)

        # ------------------------------------------------------------------------------------------------
        # acquire radar heatmap
        folder_index = fname.split('-')[0].split('_')[-1]
        radar_heatmap_index = fname.split('-')[1]
        radar_heatmap_fullname = f'{folder_index}/radar_rea_cube/{radar_heatmap_index}.bin'
        print('radar_heatmap_fullname: ', radar_heatmap_fullname)
        radar_heatmap_path_ = os.path.join(radar_heatmap_path, radar_heatmap_fullname)
        radar_heatmap = np.fromfile(radar_heatmap_path_, dtype=np.float32).reshape(256, 32, 128)
        radar_rea_heatmap = torch.from_numpy(radar_heatmap.mean(axis=1).squeeze()).float()
        radar_rea_heatmap = torch.pow(2**radar_rea_heatmap, 0.5)
        radar_rea_heatmap_unsqueezed = radar_rea_heatmap.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        radar_rea_heatmap_interpolated = F.interpolate(radar_rea_heatmap_unsqueezed, size=(512, 768), mode='bilinear', align_corners=False)
        radar_rea_heatmap_interpolated = radar_rea_heatmap_interpolated.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        
        radar_ra_heatmap = polar_to_cartesian(radar_rea_heatmap_interpolated)
        radar_heatmap_list.append(radar_ra_heatmap)

images_list = []
for fname in valid_fname_list:
    fname = fname.split('_')[-1]
    images_file = os.path.join(L12_points_path, '/scratch/project/cpautodriving/yanlongyang/kradar/data/cam-front', f'cam-front_{fname}.png')
    print(images_file)
    if os.path.exists(images_file):
        print('Loading image: ', images_file)
        images = cv2.imread(images_file)
        images_list.append(images[:,0:1279,:])
        print(images.shape)

cfar_radar_points_list = []
for fname in valid_fname_list:
    cfar_radar_points_file = os.path.join(L12_points_path, 'CFAR/radar_points', f'{fname}.bin')
    if os.path.exists(cfar_radar_points_file):
        cfar_radar_points = np.fromfile(cfar_radar_points_file, dtype=np.float32).reshape(-1, 3)
        cfar_radar_points_list.append(cfar_radar_points)

l2_radar_points_list = []
for fname in valid_fname_list:
    radar_points_file = os.path.join(L12_points_path, 'L2_Reg/radar_points', f'{fname}.bin')
    if os.path.exists(radar_points_file):
        radar_points = np.fromfile(radar_points_file, dtype=np.float32).reshape(-1, 3)
        l2_radar_points_list.append(radar_points)

l1_radar_points_list = []
for fname in valid_fname_list:
    radar_points_file = os.path.join(L12_points_path, 'L1_Reg/radar_points', f'{fname}.bin')
    if os.path.exists(radar_points_file):
        radar_points = np.fromfile(radar_points_file, dtype=np.float32).reshape(-1, 3)
        l1_radar_points_list.append(radar_points)

radarHD_points_list = []
for fname in valid_fname_list:
    radarHD_points_file = os.path.join('/scratch/project/cpautodriving/yanlongyang/dataset_with_labels/radarHD_models/0919_RADIal_radarHD_UNet/radar_points', f'{fname}.bin')
    if os.path.exists(radarHD_points_file):
        radarHD_points = np.fromfile(radarHD_points_file, dtype=np.float32).reshape(-1, 3)
        radarHD_points = radarHD_points[::16]
        radarHD_points_list.append(radarHD_points)

diffradar_points_list = []
for fname in valid_fname_list:
    diffradar_points_file = os.path.join('/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/nanoDiffusion/eval_kradar/radar_points', f'{fname}.bin')
    if os.path.exists(diffradar_points_file):
        diffradar_points = np.fromfile(diffradar_points_file, dtype=np.float32).reshape(-1, 3)
        diffradar_points_list.append(diffradar_points)

our_radar_points_list = []
for fname in valid_fname_list:
    radar_points_file = os.path.join(L12_points_path, '/scratch/project/cpautodriving/yanlongyang/dataset_with_labels/results_kradar/ours_valid/selected/radar_points', f'{fname}.bin')
    if os.path.exists(radar_points_file):
        radar_points = np.fromfile(radar_points_file, dtype=np.float32).reshape(-1, 3)
        our_radar_points_list.append(radar_points)

print('length of lidar_points_list: ', len(lidar_points_list))
print('length of images_list: ', len(images_list))
print('length of cfar_radar_points_list: ', len(cfar_radar_points_list))
print('length of l1_radar_points_list: ', len(l1_radar_points_list))
print('length of l2_radar_points_list: ', len(l2_radar_points_list))
print('length of radarHD_points_list: ', len(radarHD_points_list))
print('length of diffradar_points_list: ', len(diffradar_points_list))
print('length of our_radar_points_list: ', len(our_radar_points_list))

cnt = 0
# fid_scores = []
chamfer_distances = []
ucd_scores = []
mhd_scores = []
umhd_scores = []

for images, radar_heatmap, lidar_points, cfar_radar_points, l1_radar_points, l2_radar_points, radarHD_points, diffradar_points, our_radar_points in zip(images_list, radar_heatmap_list, lidar_points_list, cfar_radar_points_list, l1_radar_points_list, l2_radar_points_list, radarHD_points_list, diffradar_points_list, our_radar_points_list):
    plt.figure(figsize=(21, 3), facecolor='white')

    plt.subplot(1, 8, 2)
    plt.imshow(radar_heatmap, aspect='auto')  # Adjust aspect ratio to fit the height
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.tight_layout()

    # Calculate metrics for each radar points set
    metrics = {
        'cfar': calculate_metrics(cfar_radar_points, lidar_points),
        'l1': calculate_metrics(l1_radar_points, lidar_points),
        'l2': calculate_metrics(l2_radar_points, lidar_points),
        'radarHD': calculate_metrics(radarHD_points, lidar_points),
        'diffradar': calculate_metrics(diffradar_points, lidar_points),
        'our': calculate_metrics(our_radar_points, lidar_points)
    }

    # Store FID, Chamfer distances, UCD, MHD, and UMHD for averaging later
    # fid_scores.append([metrics['cfar']['fid'], metrics['l1']['fid'], metrics['l2']['fid'], metrics['radarHD']['fid'], metrics['diffradar']['fid'], metrics['our']['fid']])
    chamfer_distances.append([metrics['cfar']['cd'], metrics['l1']['cd'], metrics['l2']['cd'], metrics['radarHD']['cd'], metrics['diffradar']['cd'], metrics['our']['cd']])
    ucd_scores.append([metrics['cfar']['ucd'], metrics['l1']['ucd'], metrics['l2']['ucd'], metrics['radarHD']['ucd'], metrics['diffradar']['ucd'], metrics['our']['ucd']])
    mhd_scores.append([metrics['cfar']['mhd'], metrics['l1']['mhd'], metrics['l2']['mhd'], metrics['radarHD']['mhd'], metrics['diffradar']['mhd'], metrics['our']['mhd']])
    umhd_scores.append([metrics['cfar']['umhd'], metrics['l1']['umhd'], metrics['l2']['umhd'], metrics['radarHD']['umhd'], metrics['diffradar']['umhd'], metrics['our']['umhd']])

    plt.subplot(1, 8, 3)
    plt.scatter(lidar_points[:, 1], lidar_points[:, 0], s=0.3, c='blue', alpha=1.0)  
    plt.scatter(cfar_radar_points[:, 1], cfar_radar_points[:, 0], s=0.01, c='red', alpha=1.0)  
    plt.tight_layout()
    plt.axis('off')

    plt.subplot(1, 8, 4)
    plt.scatter(lidar_points[:, 1], lidar_points[:, 0], s=0.3, c='blue', alpha=1.0)  
    plt.scatter(l1_radar_points[:, 1], l1_radar_points[:, 0], s=0.01, c='red', alpha=0.5)  
    plt.tight_layout()
    plt.axis('off')

    plt.subplot(1, 8, 5)
    plt.scatter(lidar_points[:, 1], lidar_points[:, 0], s=0.3, c='blue', alpha=1.0)  
    plt.scatter(l2_radar_points[:, 1], l2_radar_points[:, 0], s=0.01, c='red', alpha=0.5)  
    plt.tight_layout()
    plt.axis('off')

    plt.subplot(1, 8, 6)
    plt.scatter(lidar_points[:, 1], lidar_points[:, 0], s=0.3, c='blue', alpha=1.0)  
    plt.scatter(radarHD_points[:, 1], radarHD_points[:, 0], s=0.01, c='red', alpha=1.0)  
    plt.tight_layout()
    plt.axis('off')

    plt.subplot(1, 8, 7)
    plt.scatter(lidar_points[:, 1], lidar_points[:, 0], s=0.3, c='blue', alpha=1.0)  
    plt.scatter(diffradar_points[:, 1], diffradar_points[:, 0], s=0.01, c='red', alpha=1.0)  
    plt.tight_layout()
    plt.axis('off')

    plt.subplot(1, 8, 8)
    plt.scatter(lidar_points[:, 1], lidar_points[:, 0], s=0.3, c='blue', alpha=1.0)  
    plt.scatter(our_radar_points[:, 1], our_radar_points[:, 0], s=0.01, c='red', alpha=1.0)  
    plt.tight_layout()
    plt.axis('off')

    plt.savefig(f'radar_points_comparison_{cnt}.png', bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white', edgecolor='white')
    # plt.show()
    plt.close()
    cnt += 1

# Calculate and print average metrics
# average_fid = np.mean(fid_scores, axis=0)
average_cd = np.mean(chamfer_distances, axis=0)
average_ucd = np.mean(ucd_scores, axis=0)
average_mhd = np.mean(mhd_scores, axis=0)
average_umhd = np.mean(umhd_scores, axis=0)

print("Point Cloud Categories:")
print("CFAR Radar Points, L1 Radar Points, L2 Radar Points, Radar HD Points, Difference Radar Points, Our Radar Points")

# print("Average FID scores:", average_fid)
print("Average Chamfer Distances:", average_cd)
print("Average UCD scores:", average_ucd)
print("Average MHD scores:", average_mhd)
print("Average UMHD scores:", average_umhd)
