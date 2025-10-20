import torch
import numpy as np
# import scipy.io as scio
import os
import matplotlib.pyplot as plt
from scipy import linalg
import torch.nn.functional as F

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

def otsu_threshold(image, bins=256):
    """
    Calculate optimal threshold using Otsu's method
    
    Args:
        image: Input image array
        bins: Number of histogram bins
    
    Returns:
        optimal_threshold: Optimal threshold value
    """
    # Calculate histogram
    hist, bin_edges = np.histogram(image.flatten(), bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate total pixels
    total_pixels = np.sum(hist)
    
    max_variance = 0
    optimal_threshold = 0

    # Iterate through all possible thresholds
    sum_i = 0
    sum_p = np.sum(np.multiply(hist, bin_centers))
    
    for i in range(bins-1):
        # Foreground pixels
        w0 = np.sum(hist[:i+1])
        if w0 == 0:
            continue
            
        # Background pixels    
        w1 = total_pixels - w0
        if w1 == 0:
            break
            
        sum_i += bin_centers[i] * hist[i]
        # Mean foreground intensity
        u0 = sum_i / w0
        # Mean background intensity 
        u1 = (sum_p - sum_i) / w1
        
        # Calculate between-class variance
        variance = w0 * w1 * ((u0 - u1) ** 2)
        
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = bin_centers[i]

    return optimal_threshold


def save_points_radial(sample, fname, folder_path, is_lidar=False, thresh=None):
    
    # radar coordinate
    AoA_mat = np.load('CalibrationTable.npy',allow_pickle=True).item()
    azimuth_coord = AoA_mat['Azimuth_table']
    # 如果两个维度差很多，那么要把小的插成大的
    if len(azimuth_coord)- sample.shape[-1] > 100:
        azimuth_coord = np.linspace(azimuth_coord[0], azimuth_coord[-1], sample.shape[-1])
    # 预测
    # out_ra = torch.sigmoid(1000*(sample-0.01))
    out_ra = sample
    if len(out_ra.shape) >= 3:
        out_ra_np = out_ra.mean(dim=1).detach().cpu().numpy()
        out_ra_np = np.squeeze(out_ra_np[0,...])
    else:
        out_ra_np = out_ra.detach().cpu().numpy()

    if thresh is None:
        # thresh = 0.01
        thresh = otsu_threshold(out_ra_np)
    pixel_indices = np.where(out_ra_np > thresh)
    real_coords = []
    range_coord = np.linspace(0,out_ra_np.shape[0],out_ra_np.shape[0])/out_ra_np.shape[0]*103
    for i in range(len(pixel_indices[0])):
        pixel_y, pixel_x = pixel_indices[0][i], pixel_indices[1][i]  # y corresponds to range, x to azimuth
        if pixel_x >= len(azimuth_coord) or pixel_y >= len(range_coord):
            continue
        azimuth = np.deg2rad(azimuth_coord[pixel_x])  # Convert pixel x to azimuth in radians
        range_val = range_coord[pixel_y]  # Convert pixel y to range
        x = range_val * np.cos(azimuth)  # Convert polar to Cartesian coordinates
        y = range_val * np.sin(azimuth)  # Convert polar to Cartesian coordinates
        real_coords.append([x, y, 0])  # Assuming 0 for the z-coordinate
    real_coords = np.array(real_coords).astype(np.float32)
    if is_lidar:
        out_folder = folder_path+'/lidar_points'
    else:
        out_folder = folder_path+'/radar_points'
    os.makedirs(out_folder, exist_ok=True)
    if isinstance(fname, str):
        pcd_filename = os.path.join(out_folder, fname+'.bin')
    else:
        pcd_filename = os.path.join(out_folder, f'{fname:06d}.bin')
    # pcd_filename = os.path.join(out_folder, str(fname)+'.bin')
    print(pcd_filename)
    # if os.path.exists(pcd_filename):
    #     return
    plt.figure(figsize=(10, 5), dpi=300, facecolor='black')
    plt.scatter(real_coords[:, 0], real_coords[:, 1], s=0.1, c='white')
    print('max and min of x: ', real_coords[:, 0].max(), real_coords[:, 0].min())
    print('max and min of y: ', real_coords[:, 1].max(), real_coords[:, 1].min())
    # plt.xlim(-50, 50)
    plt.xlim(0, 120)
    plt.axis('off')    # Turn off axis labels and ticks
    plt.savefig(os.path.join(out_folder, str(fname)+'.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
    if len(real_coords) > 0:
        real_coords.tofile(pcd_filename)
        print(' Frame {} bin file saved!!!!!'.format(fname))
    # save raw data
    np.save(os.path.join(out_folder, str(fname)+'.npy'), out_ra_np)
    return real_coords


def save_points_kradar(sample, fname, folder_path, is_lidar=False):
    sample = sample.squeeze()
    azimuth_coord = np.linspace(-53, 53, sample.shape[-1])
    # 预测
    # out_ra = torch.sigmoid(1000*(sample-0.01))
    out_ra = sample
    if len(out_ra.shape) >= 3:
        out_ra = out_ra.mean(dim=0)
    if isinstance(out_ra, torch.Tensor) and out_ra.device.type == 'cuda':
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
    if is_lidar:
        out_folder = folder_path+'/lidar_points'
    else:
        out_folder = folder_path+'/radar_points'
    os.makedirs(out_folder, exist_ok=True)
    # pcd_filename = os.path.join(out_folder, '{:06d}.bin'.format(fname))
    pcd_filename = os.path.join(out_folder, str(fname)+'.bin')
    print(pcd_filename)
    # if os.path.exists(pcd_filename):
    #     return
    plt.figure(figsize=(10, 5), dpi=300, facecolor='black')
    plt.scatter(real_coords[:, 0], real_coords[:, 1], s=0.1, c='white')
    print('max and min of x: ', real_coords[:, 0].max(), real_coords[:, 0].min())
    print('max and min of y: ', real_coords[:, 1].max(), real_coords[:, 1].min())
    # plt.xlim(-50, 50)
    plt.xlim(0, 120)
    plt.axis('off')    # Turn off axis labels and ticks
    plt.savefig(os.path.join(out_folder, str(fname)+'.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
    if len(real_coords) > 0:
        # add noise
        noise_level = 1.5  # Adjust the noise level as needed
        noise = np.random.normal(0, noise_level, real_coords.shape)  # Generate noise around existing points
        noisy_coords = real_coords + noise  # Add noise to the original coordinates
        real_coords = np.vstack((real_coords, noisy_coords))  # Combine original and noisy coordinates
        # save
        real_coords.tofile(pcd_filename)
        print(' Frame {} bin file saved!!!!!'.format(fname))

    # save raw data
    np.save(os.path.join(out_folder, str(fname)+'.npy'), out_ra_np)

    return real_coords

def polar_to_cartesian(out_ra_np, range_coord=None, azimuth_coord=None, x_coord=None, y_coord=None):
    out_ra_np = out_ra_np.mean(dim=1).squeeze()
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


def music(out_ra_np, num_sources=10, num_snapshots=1):
    """
    Implement the MUSIC (MUltiple SIgnal Classification) algorithm for super-resolution.
    
    Args:
    out_ra_np (numpy.ndarray): Input radar data in range-azimuth format.
    num_sources (int): Number of signal sources to detect.
    num_snapshots (int): Number of snapshots to use for covariance matrix estimation.
    
    Returns:
    numpy.ndarray: MUSIC spectrum (pseudo-spectrum)
    """
    # Ensure input is numpy array
    out_ra_np = out_ra_np.mean(dim=1).squeeze()
    out_ra_np = np.array(out_ra_np)
    
    # Get dimensions
    num_range, num_azimuth = out_ra_np.shape
    
    # Initialize MUSIC spectrum
    music_spectrum = np.zeros((num_range, num_azimuth))
    # Generate steering vectors
    steering_vectors = np.exp(-1j * np.outer(np.arange(num_azimuth), np.linspace(0, np.pi, 180)))
    
    # Iterate over range dimension
    for r in range(num_range):
        # Extract data for current range
        range_data = out_ra_np[r, :]
        
        # Reshape data for covariance matrix estimation
        data_matrix = range_data.reshape(-1, 1)
        
        # Estimate covariance matrix
        R = np.dot(data_matrix, data_matrix.conj().T) / num_snapshots
        
        # Eigendecomposition of covariance matrix
        eigenvalues, eigenvectors = linalg.eigh(R)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Separate signal and noise subspaces
        noise_eigenvectors = eigenvectors[:, num_sources:]
        
        import pdb;pdb.set_trace()
        # Compute MUSIC spectrum for current range
        for i in range(num_azimuth):
            a = steering_vectors[:, i].reshape(-1, 1)
            music_spectrum[r, i] = 1 / np.abs(a.conj().T @ noise_eigenvectors @ noise_eigenvectors.conj().T @ a)
    
    return music_spectrum


if __name__ == '__main__':
    import argparse
    from model_loader import load_yaml
    from data.dataloader import get_dataset, get_dataloader

    save_folder = './results_kradar_subspace'
    os.makedirs(save_folder, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', default="configs/tasks/kradar_imaging_config.yaml", type=str)
    args = parser.parse_args()
    task_config = load_yaml(args.task_config)

    # Prepare dataloader
    data_config = task_config['data']
    dataset = get_dataset(**data_config) #, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    for i, data in enumerate(loader):

        lidar = data['lidar_ra'].float()/data['lidar_ra'].max()
        radar = data['radar_ra'].float()/data['radar_ra'].max()
        fname = data['data_fname'][0]


        # save radar_cate
        radar_cate = polar_to_cartesian(radar)
        plt.figure()
        plt.imshow(radar_cate, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_folder, str(fname)+'_radar.png'))
        plt.close()
        print('saved {}'.format(fname))


        # save lidar_cate
        lidar_cate = polar_to_cartesian(lidar)
        plt.figure()
        plt.imshow(lidar_cate, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_folder, str(fname)+'_lidar.png'))
        plt.close()
        print('saved {}'.format(fname))

        # save lidar points
        save_points_kradar(lidar, fname, save_folder, is_lidar=True)

        # save radar points
        save_points_kradar(radar, fname, save_folder, is_lidar=False)

        # save music
        music_spec = np.log10(music(radar))
        music_spec = (music_spec - music_spec.min())/(music_spec.max()-music_spec.min())
        print('music mean, max, min: ', music_spec.mean(), music_spec.max(), music_spec.min())
        plt.figure()
        plt.imshow(music_spec[:, 10:], cmap='gray')
        # for r in range(music_spec.shape[0]):
        #     plt.plot(music_spec[r, :])
        plt.axis('off')
        plt.savefig(os.path.join(save_folder, str(fname)+'_music.png'))
        plt.close()

        break


# export PYTHONPATH=$PYTHONPATH:$(pwd); python util/save_points.py --task_config configs/tasks/kradar_imaging_config.yaml
