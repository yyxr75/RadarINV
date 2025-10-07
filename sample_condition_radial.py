from ldm_inverse.condition_methods import get_conditioning_method
from ldm.models.diffusion.ddim import DDIMSampler
from scripts.utils import clear_color, mask_generator
import matplotlib.pyplot as plt
from ldm_inverse.measurements import get_noise, get_operator
from functools import partial
import numpy as np
from model_loader import load_model_from_config, load_yaml
import os
import torch
import torchvision.transforms as transforms
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from skimage.metrics import peak_signal_noise_ratio as psnr
from util.save_points import save_points_radial
from torchvision.utils import save_image

def get_model(args, device=None):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_ckpt, device, train=False)

    return model

def load_data(fname):
    data = np.fromfile(fname).reshape(512, 768)
    radar = torch.from_numpy(data).squeeze().unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 768)
    return radar

def create_folder(args, fname):
    out_path = os.path.join(args.save_dir)
    folder_of_params = os.path.join(out_path, fname)
    if not os.path.exists(folder_of_params):
        os.makedirs(folder_of_params)
    # Save the configuration file
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(folder_of_params, img_dir), exist_ok=True)
    return folder_of_params

parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)

parser.add_argument("--input_image", type=str, default='demo_data/lidar_mask_000181.bin')
parser.add_argument('--ldm_config', default="configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml", type=str)
parser.add_argument('--diffusion_ckpt', default="models/ldm/epoch=000098.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/radial_imaging_config.yaml", type=str)

parser.add_argument('--simulation', action='store_true', help='If true, simulate the measurement; Otherwise, load the real measurement')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='results_radial_radarINV')
parser.add_argument('--ddim_steps', default=1000, type=int)
parser.add_argument('--ddim_eta', default=0.1, type=float) 
parser.add_argument('--n_samples_per_class', default=1, type=int)
parser.add_argument('--ddim_scale', default=1.0, type=float)

parser.add_argument('--step_size_dynamic', default=0.001, type=float)
parser.add_argument('--step_size_static', default=None, type=float)
parser.add_argument('--measurement_scale', default=1, type=float)
parser.add_argument('--measurement_step_number', default=15, type=int)

parser.add_argument('--resample_sigma', default=80, type=float)
parser.add_argument('--save_process', action='store_true', help='Save intermediate process results')

parser.add_argument('--test_var', default=0, type=int)
parser.add_argument('--unet_lr', default=0.001, type=float)
parser.add_argument('--unet_iters', default=10, type=int)


args = parser.parse_args()

# ------------------------------------------------------------
# initialize
# ------------------------------------------------------------
# Load configurations
task_config = load_yaml(args.task_config)
# Device setting
device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
print(f"Device set to {device_str}.")
device = torch.device(device_str) 
# Loading model
model = get_model(args, device=device)
sampler = DDIMSampler(model) # Sampling using DDIM

# Prepare Operator and noise
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning
print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

constraint_method = None

# Instantiating sampler
sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward, constraint_fn=constraint_method,
                                        S=args.ddim_steps,
                                        cond_method=task_config['conditioning']['main_sampler'],
                                        conditioning=None,
                                        ddim_use_original_steps=True,
                                        batch_size=args.n_samples_per_class,
                                        shape=[4, 32, 48], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta,
                                        # noise_dropout=0.1,
                                        **vars(args))

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
    mask_gen = mask_generator(**measure_config['mask_opt'])


from util.rpl import RadarSignalProcessing
RSP = RadarSignalProcessing('CalibrationTable.npy',method='RA',device='cpu') #,lib='PyTorch')
# ------------------------------------------------------------
# Do inference
# ------------------------------------------------------------
import random

# Set a seed for reproducibility
random.seed(55)

try:
    data = load_data(args.input_image)
except:
    import pdb;pdb.set_trace()
    pass
fname = args.input_image.split('/')[-1].split('.')[0] 

folder_of_params = create_folder(args, fname)

print('***************************go with {}   ***************************'.format(fname))
if args.simulation:
    y_n = operator.forward(data.to(device))
else:
    y_n = data.to(device)

ref_img = torch.randn(y_n.shape).to(device) # useless

# save label
save_image(data, os.path.join(folder_of_params, 'label', str(fname)+'_label.png'), normalize=True)
# save y_n
save_image(y_n, os.path.join(folder_of_params, 'input', str(fname)+'_input.png'), normalize=True)
# inference
samples_ddim, _ = sample_fn(original=ref_img, measurement=y_n, fname=fname, test_var=None, folder_of_params=folder_of_params) #, constraint_fn=None, conditioning=cond)
x_samples_ddim = model.decode_first_stage(samples_ddim.detach())
# save points
save_points_radial(x_samples_ddim, fname, folder_of_params, is_lidar=False)
# save recon
save_image(x_samples_ddim, os.path.join(folder_of_params, 'recon', str(fname)+'_recon.png'), normalize=True)

# python sample_condition_radial.py --step_size_dynamic 0.001  --measurement_scale 1.0 --measurement_step_number 20 --unet_lr 0.001 --unet_iters 10 --resample_sigma 80 --save_process --gpu 1