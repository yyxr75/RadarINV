'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import torch

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
        H,W = shape[0], shape[1] # coloradar 中的数据大小
        self.y_coords, self.x_coords = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device))
        # Initialize any other necessary components here
        self.num_antennas = num_antennas
        self.wavelength = 0.03
        self.d = self.wavelength / 2

    def cond_fn(self, img, ref_img=None, threshold=0.01, sharpness=1000):
        mask = torch.sigmoid(sharpness * (img.mean(dim=1, keepdim=True) - threshold)) #这是一个近似，可能导致效果不行
        N,C,H,W = img.shape
        x_coords = self.x_coords
        azimuth_grid = x_coords*150/W-75 # radial angle setting
        theta_grid =  torch.deg2rad(azimuth_grid)
        phase_diff = 2 * np.pi * self.d * torch.sin(theta_grid) / self.wavelength
        antennas = torch.arange(self.num_antennas, device=self.device)
        antennas = antennas.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, self.num_antennas, H, W)
        signals = torch.exp(1j * antennas* phase_diff)

        signals = signals * mask

        fft_result = torch.fft.fft(signals, dim=1, n=W)

        fft_result = torch.fft.fftshift(fft_result, dim=1)
        output_range_azimuth = torch.abs(fft_result.mean(dim=-1)).permute(0, 2, 1)
        output = output_range_azimuth.unsqueeze(1).repeat(1, 3, 1, 1)

        return output

    def forward(self, img, ref_img=None, num_virtual_antennas=86, max_distance=103, num_distance_bins=512, num_angle_bins=768, **kwargs):
        return self.cond_fn(img, ref_img)

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
