from ldm_inverse.condition_methods import get_conditioning_method
from ldm_inverse.radarHD import get_constraint_method
from ldm.models.diffusion.ddim import DDIMSampler
from data.dataloader import get_dataset, get_dataloader
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

def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_ckpt)

    return model

def create_folder(args, fname, index=None):
    out_path = os.path.join(args.save_dir)
    params = 'test_file_number_{}_step_size_dynamic_{}_step_size_static_{}_measurement_scale_{}_measurement_step_number_{}_fname_{}_index_{}'.format(args.test_filenumber, args.step_size_dynamic, args.step_size_static, args.measurement_scale, args.measurement_step_number, fname, index)
    folder_of_params = os.path.join(out_path, params)
    if not os.path.exists(folder_of_params):
        os.makedirs(folder_of_params)
    # Save the configuration file
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(folder_of_params, img_dir), exist_ok=True)
    return folder_of_params

parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)

parser.add_argument('--ldm_config', default="configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml", type=str)
parser.add_argument('--diffusion_ckpt', default="models/ldm/radial_ldm_countryside_scene_epoch=000131.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/radial_imaging_config.yaml", type=str)

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='results_radial_radarINV')
parser.add_argument('--ddim_steps', default=1000, type=int)
parser.add_argument('--ddim_eta', default=0.1, type=float) 
parser.add_argument('--n_samples_per_class', default=1, type=int)
parser.add_argument('--ddim_scale', default=1.0, type=float)

# scan parameters
parser.add_argument('--test_filenumber', default=1, type=int)
parser.add_argument('--step_size_dynamic', default=0.001, type=float)
parser.add_argument('--step_size_static', default=None, type=float)
parser.add_argument('--measurement_scale', default=2.5, type=float)
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
device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
print(f"Device set to {device_str}.")
device = torch.device(device_str)  

# Loading model
model = get_model(args)
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

# Prepare dataloader
data_config = task_config['data']
dataset = get_dataset(**data_config)
loader = get_dataloader(dataset, batch_size=args.n_samples_per_class, num_workers=0, train=False)

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
    mask_gen = mask_generator(**measure_config['mask_opt'])


from ldm_inverse.visualize import visualizer
Visualizer = visualizer()
from util.rpl import RadarSignalProcessing
RSP = RadarSignalProcessing('CalibrationTable.npy',method='RA',device='cpu') #,lib='PyTorch')
# ------------------------------------------------------------
# Do inference
# ------------------------------------------------------------
test_filenumber = args.test_filenumber
import random

# Set a seed for reproducibility
random.seed(55)

for i, data_dict in enumerate(loader):

    lidar = data_dict['lidar_ra'].to(device)
    radar = data_dict['radar_ra'].to(device)
    fname = data_dict['data_fname'].item()

    folder_of_params = create_folder(args, fname, index=i)
    ref_img = lidar
    # save points
    save_points_radial(ref_img, fname, folder_of_params, is_lidar=True)

    print('***************************go with {}   ***************************'.format(fname))
    y_n = radar

    # input
    input_img = ref_img.mean(dim=1).detach().cpu().numpy().squeeze()
    plt.imshow(input_img, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(folder_of_params, 'input', str(fname)+'_input.png'), bbox_inches='tight', pad_inches=0, dpi=600)
    # label
    label_img = y_n.mean(dim=1).detach().cpu().numpy().squeeze()
    plt.imshow(label_img, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(folder_of_params, 'label', str(fname)+'_label.png'), bbox_inches='tight', pad_inches=0, dpi=600)
    # inference
    samples_ddim, _ = sample_fn(original=ref_img, measurement=y_n, fname=fname, test_var=None, folder_of_params=folder_of_params) #, constraint_fn=None, conditioning=cond)
    x_samples_ddim = model.decode_first_stage(samples_ddim.detach())
    # save points
    save_points_radial(x_samples_ddim, fname, folder_of_params, is_lidar=False)
    # recon
    output_img = x_samples_ddim
    plt.imshow(output_img.mean(dim=1).detach().cpu().numpy().squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(folder_of_params, 'recon', str(fname)+'_recon.png'), bbox_inches='tight', pad_inches=0, dpi=600)
    # break

# python sample_condition_radial.py --test_filenumber 5 --step_size_dynamic 0.001  --measurement_scale 1.0 --measurement_step_number 20 --unet_lr 0.001 --unet_iters 10 --resample_sigma 80 --save_process --gpu 0