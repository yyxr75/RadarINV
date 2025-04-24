'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel
import numpy as np

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m
from ldm_inverse.unet import UNet

import matplotlib.pyplot as plt
# Save output as an image
import os

# import sys
# sys.path.append('/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning')
# from RADIal.SignalProcessing.rpl import RadarSignalProcessing
# RSP = RadarSignalProcessing('/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/RADIal/SignalProcessing/CalibrationTable.npy',method='RA',device='cpu') #,lib='PyTorch')

#from torch_radon import Radon, solvers


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        data = data.to(self.device) # Sending to device
        return self.down_sample(data).to(self.device) # Sending to device

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        data = data.to(self.device) # Sending to device
        return self.conv(data).to(self.device)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude


@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        
#         random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        np.random.seed(0)
        kernel_np = np.random.randn(1,512,2,2)*1.2
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)

# ===============
# radar imaging process
# ===============

@register_operator(name='radar_imaging')
class RadarImagingOperator(LinearOperator):
    def __init__(self, num_antennas, device, shape):
        # path_calib_mat = '/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/RADIal/SignalProcessing/CalibrationTable.npy'
        self.device = device
        self.num_antennas = num_antennas
        # self.AoA_mat = np.load(path_calib_mat,allow_pickle=True).item()
        # 定义固定的坐标grid
        # H,W = 512, 768
        # H,W = 256, 64 # radarHD 中的数据大小
        H,W = shape[0], shape[1] # coloradar 中的数据大小
        self.y_coords, self.x_coords = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device))
        # Initialize any other necessary components here
        self.num_antennas = num_antennas
        self.wavelength = 0.03
        self.d = self.wavelength / 2
        '''
        # load unet as condition function
        self.unet = UNet(3, 3).to(self.device)
        model_path = '/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/resample/models/cond_models/080.pt_gen'
        checkpoint = torch.load(model_path, map_location=self.device)
        self.unet.load_state_dict(checkpoint['state_dict'])
        '''
    
    def add_noise(self, data, noise_type='gaussian', **kwargs):
        if torch.is_complex(data):
            if noise_type == 'gaussian':
                mean = kwargs.get('mean', 0)
                std = kwargs.get('std', 1)
                noise_real = torch.randn(data.size(), dtype=torch.float32) * std + mean
                noise_imag = torch.randn(data.size(), dtype=torch.float32) * std + mean
                noise = torch.complex(noise_real, noise_imag).to(self.device)
            elif noise_type == 'poisson':
                noise_real = torch.poisson(data.real)
                noise_imag = torch.poisson(data.imag)
                noise = torch.complex(noise_real, noise_imag).to(self.device)
            else:
                raise ValueError("Unsupported noise type")
        else:
            if noise_type == 'gaussian':
                mean = kwargs.get('mean', 0)
                std = kwargs.get('std', 1)
                noise = torch.randn(data.size(), dtype=torch.float32) * std + mean
                noise = noise.to(self.device)
            elif noise_type == 'poisson':
                noise = torch.poisson(data)
                noise = noise.to(self.device)
            else:
                raise ValueError("Unsupported noise type")
        return data + noise

    def cfar_2d(self, data, guard_cells=2, training_cells=4, false_alarm_rate=0.001):
        """
        Perform 2D CFAR on the input data using tensor operations to maintain gradients.
        """
        N, C, H, W = data.shape
        pad_size = training_cells + guard_cells
        padded_data = F.pad(data, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)

        # Create a mask to exclude guard cells and the cell under test
        mask = torch.ones((2 * training_cells + 2 * guard_cells + 1, 2 * training_cells + 2 * guard_cells + 1), device=data.device)
        mask[training_cells:training_cells + 2 * guard_cells + 1, training_cells:training_cells + 2 * guard_cells + 1] = 0
        # Compute the noise level using convolution
        noise_level = F.conv2d(padded_data, mask.view(1, 1, *mask.shape).expand(C, -1, -1, -1), groups=C) / mask.sum()

        # Calculate the threshold
        threshold = noise_level * false_alarm_rate

        # Compare the cell under test with the threshold
        output = data * (data > threshold).float()
        return output, threshold


    def cond_fn_unet(self, img, ref_img=None):
        return self.unet(img)

    # ref_img # 测试2024-07-09，为了更好计算成像，计算理论成像结果与实际的差值
    def cond_fn(self, img, ref_img=None, threshold=0.01, sharpness=1000):
        # sharpness = 1000是因为threshold太小，导致sigmoid需要很sharp，mask相当于是真实的目标点位置
        # threshold = img.mean(dim=1, keepdim=True)+0.01
        mask = torch.sigmoid(sharpness * (img.mean(dim=1, keepdim=True) - threshold)) #这是一个近似，可能导致效果不行
        N,C,H,W = img.shape
        # plt.figure(figsize=(10, 10), dpi=100)
        # plt.imshow(mask.detach().cpu().numpy().squeeze(), cmap='gray')
        # plt.axis('off')
        # plt.savefig('mask.png', bbox_inches='tight', pad_inches=0)
        
        # ------------------
        # 扩展像素坐标grid，计算每一个像素位置的真实空间坐标
        # ------------------
        # y_coords = self.y_coords.unsqueeze(0).expand(N, -1, -1)
        x_coords = self.x_coords
        # azimuth_grid = x_coords*180/W-90 # coloradar angle setting
        azimuth_grid = x_coords*150/W-75 # radial angle setting
        # range_grid = y_coords*103/512
        # ------------------
        # 计算每一个angle对应的signal(attena维)
        # ------------------
        theta_grid =  torch.deg2rad(azimuth_grid)
        # 生成每个天线的信号
        # 1. noise这里，表示信号相位有随机噪声
        phase_diff = 2 * np.pi * self.d * torch.sin(theta_grid) / self.wavelength
        # phase_diff = self.add_noise(phase_diff, noise_type='gaussian', mean=0, std=0.2)
        antennas = torch.arange(self.num_antennas, device=self.device)
        antennas = antennas.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, self.num_antennas, H, W)
        signals = torch.exp(1j * antennas* phase_diff)

        signals = signals * mask

        # 对signal的dim=1做fft和fft shift
        fft_result = torch.fft.fft(signals, dim=1, n=W)

        # plt.figure(figsize=(10, 10), dpi=100)
        # ifft_result = torch.fft.ifft(fft_result, dim=1, n=W)
        # plt.imshow(np.abs(ifft_result.mean(dim=-1).squeeze().cpu().detach().numpy()), cmap='gray')
        # plt.axis('off')
        # plt.savefig('signals_masked.png', bbox_inches='tight', pad_inches=0)

        fft_result = torch.fft.fftshift(fft_result, dim=1)
        output_range_azimuth = torch.abs(fft_result.mean(dim=-1)).permute(0, 2, 1)
        output = output_range_azimuth.unsqueeze(1).repeat(1, 3, 1, 1)

        return output

    def cond_fn_noisy(self, img, ref_img=None, threshold=0.01, sharpness=1000):
        # Sigmoid mask as before
        mask = torch.sigmoid(sharpness * (img.mean(dim=1, keepdim=True) - threshold))
        N, C, H, W = img.shape
        # Coordinate setup
        x_coords = self.x_coords
        azimuth_grid = x_coords * 150 / 768 - 75
        theta_grid = torch.deg2rad(azimuth_grid)

        # Phase difference calculation
        phase_diff = 2 * np.pi * self.d * torch.sin(theta_grid) / self.wavelength

        # Add phase noise
        # phase_noise = torch.randn_like(phase_diff) * 0.1  # Adjust std for noise level
        # phase_diff += phase_noise

        # Generate antenna signals with non-linearity and multipath
        antennas = torch.arange(self.num_antennas, device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, self.num_antennas, H, W)
        signals = torch.exp(1j * antennas * phase_diff)

        # Add non-linearity
        signals = torch.tanh(signals.real) + 1j * torch.tanh(signals.imag)

        # Multipath effect (simplified)
        multipath_delay = torch.rand_like(signals) * 0.5  # Random delay
        multipath_signal = 0.3 * torch.exp(1j * antennas * (phase_diff + multipath_delay))  # 30% strength of original
        signals += multipath_signal

        # Apply mask
        signals = signals * mask

        # Apply windowing function (e.g., Hamming window)
        window = torch.hamming_window(self.num_antennas, device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        signals = signals * window

        # FFT and FFT shift
        fft_result = torch.fft.fft(signals, dim=1, n=W)
        fft_result = torch.fft.fftshift(fft_result, dim=1)
        # Calculate output
        output_range_azimuth = torch.abs(fft_result.mean(dim=-1)).permute(0, 2, 1)

        # # Add thermal noise
        # thermal_noise = torch.randn_like(output_range_azimuth) * 0.05  # Adjust std for noise level
        # output_range_azimuth += thermal_noise

        # Clip to ensure non-negative values
        output_range_azimuth = torch.clamp(output_range_azimuth, min=0)

        # Normalize output
        output_range_azimuth = (output_range_azimuth - output_range_azimuth.min()) / (output_range_azimuth.max() - output_range_azimuth.min())

        output = output_range_azimuth.unsqueeze(1).repeat(1, 3, 1, 1)

        return output

    def forward(self, img, ref_img=None, num_virtual_antennas=86, max_distance=103, num_distance_bins=512, num_angle_bins=768, **kwargs):
        return self.cond_fn(img, ref_img)
        # return self.cond_fn_unet(img, ref_img)
        # threshold = kwargs.get('threshold', 0.01)
        # return self.cond_fn_noisy(img, threshold=threshold)

    def transpose(self, data, **kwargs):
        return data
        
@register_noise(name='radar_capture')
class RadarCapture(Noise):
    def __init__(self, ):
        pass


    def forward(self, data):
        '''
        no noise
        '''

        return data

if __name__ == '__main__':
    # simple, 3t4r, ti cascade-2243, ideal-condition
    antenna_list = [4, 12, 86, 192]
    lidar_path = '/scratch/user/yanlongyang/Project1_Diffusion_related/dataset_with_labels/lidars_mask/'

    save_path = '/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/resample/results_likelihood_func/'
    cnt = 0
    for file in os.listdir(lidar_path):
        print(f'################file={file}################')
        lidar_fname = lidar_path + file
        lidar_data = np.fromfile(lidar_fname).reshape([512, 768])
        cnt += 1
        if cnt > 5:
            break

        # input_radar_path = '/scratch/user/yanlongyang/Project1_Diffusion_related/dataset_with_labels/radars_ra_interp/000018.bin'
        # radar = np.fromfile(input_radar_path).reshape([512, 768])
        radar = lidar_data


        for antenna in antenna_list:
            print(f'**************antenna={antenna}**************')
            save_folder = save_path + f'antenna={antenna}/'
            os.makedirs(save_folder, exist_ok=True)
                    
            plt.figure(figsize=(10, 10), dpi=100)
            plt.imshow(radar, cmap='gray')
            plt.axis('off')
            plt.savefig(save_folder + f'radar_imaging_visualization_raw_{file}.png', bbox_inches='tight', pad_inches=0)

            radar_cuda = torch.from_numpy(radar).unsqueeze(0).unsqueeze(0).to(torch.float32).to('cuda')
            radar_cuda = (radar_cuda-radar_cuda.min())/(radar_cuda.max()-radar_cuda.min())
            operator = RadarImagingOperator(num_antennas=antenna, device='cuda', shape=(512, 768))
            output = operator.forward(radar_cuda)

            plt.figure(figsize=(10, 10), dpi=100)
            plt.imshow(output.mean(dim=1).detach().cpu().numpy().squeeze(), cmap='gray')
            plt.axis('off')
            plt.savefig(save_folder + f'radar_imaging_visualization_{file}.png', bbox_inches='tight', pad_inches=0)

# export PYTHONPATH=$PYTHONPATH:$(pwd) && python ldm_inverse/measurements.py
