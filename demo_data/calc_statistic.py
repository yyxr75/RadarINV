import os
import numpy as np
import torch

def load_data(fname, shape=(512, 768), target_shape=None):
    data = np.fromfile(fname).reshape(shape)

    # Optional downsample
    if target_shape is not None:
        scale_y = shape[0] // target_shape[0]
        scale_x = shape[1] // target_shape[1]
        data = data[::scale_y, ::scale_x]

    radar = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
    return radar

datalist_path = '/home/icclab/Documents/yyl/RADIal_Dataset/RADIal/dataset_with_labels/radars_ra_interp'
mean_of_mean = 0
mean_of_var = 0
cnt = 0

for filename in os.listdir(datalist_path):
    if filename.endswith('.bin'):
        full_path = os.path.join(datalist_path, filename)
        data = load_data(full_path, target_shape=(128, 192))
        mean_of_mean += data.mean().item()
        mean_of_var += data.var().item()
        # print(f'Data path: {full_path}, mean: {mean:.6f}, var: {var:.6f}')
        cnt += 1
print(f'All mean: {mean_of_mean/cnt:.6f}, var: {mean_of_var/cnt:.6f}')
