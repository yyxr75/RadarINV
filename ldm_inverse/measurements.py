'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import torch
import numpy as np

def print_memory(prefix=""):
    allocated = torch.cuda.memory_allocated(1) / 1024**2
    reserved = torch.cuda.memory_reserved(1) / 1024**2
    print(f"{prefix} Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

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

# ===============
# radar imaging process
# ===============

@register_operator(name='radar_imaging')
class RadarImagingOperator(LinearOperator):
    def __init__(self, num_antennas, device, shape):
        self.device = device
        self.num_antennas = num_antennas
        
        # Assume shape is (H, W)
        H, W = shape[0], shape[1] 
        self.H = H
        self.W = W
        
        # --- 1. Coordinate and Physical Parameter Calculation ---
        # Use indexing='ij' to ensure H, W correspond to y, x
        self.y_coords, self.x_coords = torch.meshgrid(
            torch.arange(H, device=self.device), 
            torch.arange(W, device=self.device), 
            indexing='ij' # Use 'ij' to ensure y_coords corresponds to H, x_coords to W
        )
        
        self.wavelength = 0.03
        self.d = self.wavelength / 2
        
        # Phase difference calculation based on x-coordinates (W dimension)
        azimuth_grid = self.x_coords.float() * 150 / W - 75 # H x W
        self.theta_grid = torch.deg2rad(azimuth_grid)      # H x W
        
        # phase_diff shape: H x W
        self.phase_diff = (2 * np.pi * self.d * torch.sin(self.theta_grid) / self.wavelength).float()
        
        # --- 2. Prepare Antenna Indices ---
        # antennas shape: [1, NumA, 1, 1]
        antennas = torch.arange(self.num_antennas, device=self.device).float().reshape(1, self.num_antennas, 1, 1)

        # --- 3. Calculate and Store the Fixed self.signals Matrix (Memory Optimization) ---
        
        # Adjust phase_diff dimensions for broadcasting with mask and antennas
        # phase_diff_broadcastable shape: 1 x 1 x H x W
        # Detach to explicitly ensure no gradient tracking (default behavior)
        phase_diff_broadcastable = self.phase_diff.unsqueeze(0).unsqueeze(0).detach() 

        # Calculate exponent term: [1, NumA, 1, 1] * [1, 1, H, W] -> [1, NumA, H, W]
        exp_term = antennas * phase_diff_broadcastable
        
        # self.signals shape: 1 x NumA x H x W (Complex tensor)
        # This is a constant and does not require gradient tracking.
        # print_memory("Before signals creation")
        self.signals = torch.exp(1j * exp_term).contiguous().detach()
        self.signals.requires_grad_(False)
        # print_memory("After signals creation")


    def cond_fn(self, img, ref_img=None, threshold=0.01, sharpness=1000,
                h_chunk_size=32, gpu_id=1):
        """
        cond_fn with H chunking to reduce memory, preserve gradient for mask/img.
        h_chunk_size: number of range bins (H) per FFT chunk
        """
        # --- 1. mask ---
        mask = img.mean(dim=1, keepdim=True)  # N x 1 x H x W
        signals = self.signals * mask  # N x NumA x H x W

        N, NumA, H, W = signals.shape
        # output_accum shape: N x W x H  -> 因为 FFT 后天线维度补到 W
        output_accum = torch.zeros(N, W, H, device=signals.device, dtype=signals.dtype)

        # --- 2. 按 H 分块做 FFT ---
        for h_start in range(0, H, h_chunk_size):
            h_end = min(h_start + h_chunk_size, H)
            chunk = signals[:, :, h_start:h_end, :]  # N x NumA x h_chunk x W

            # FFT along antenna dimension, 补到 n=W
            fft_chunk = torch.fft.fft(chunk, dim=1, n=W)  # N x W x h_chunk x W
            fft_chunk = torch.roll(fft_chunk, shifts=W//2, dims=1)  # FFT shift

            # 对最后一维 W 做平均 -> 输出 shape: N x W x h_chunk
            mean_chunk = fft_chunk.mean(dim=-1)  # N x W x h_chunk

            # 写入累积 tensor
            output_accum[:, :, h_start:h_end] = mean_chunk

            # 释放临时 tensor
            del chunk, fft_chunk, mean_chunk
            torch.cuda.empty_cache()

        # --- 3. 输出 reshape + repeat 通道 ---
        output_range_azimuth = torch.abs(output_accum).permute(0, 2, 1)  # N x H x W
        del output_accum, signals, mask
        torch.cuda.empty_cache()
        # print(f"[MEM] After final cleanup: {torch.cuda.memory_allocated(gpu_id)/1024**2:.2f} MB")

        output = output_range_azimuth.unsqueeze(1).repeat(1, 3, 1, 1)  # N x 3 x H x W
        return output

    def forward(self, img, ref_img=None, num_virtual_antennas=86, max_distance=103, num_distance_bins=512, num_angle_bins=768, **kwargs):
        # forward method calls cond_fn
        return self.cond_fn(img, ref_img)

    def transpose(self, data, **kwargs):
        # Placeholder implementation for the transpose operation (e.g., conjugate transpose for complex operators)
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
