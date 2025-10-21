import os
import argparse
import torch
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision.utils import save_image
from ldm_inverse.condition_methods import get_conditioning_method
from ldm.models.diffusion.ddim import DDIMSampler
from scripts.utils import clear_color, mask_generator
from ldm_inverse.measurements import get_noise, get_operator
from model_loader import load_model_from_config, load_yaml
from util.save_points import save_points_radial

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
TARGET_MEAN = 0.037801 # collected from radar data
TARGET_VAR = 0.001207 # collected from radar data
TARGET_STD = np.sqrt(TARGET_VAR)

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def get_model(args, device=None):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_ckpt, device, train=False)
    return model, config

def load_data(fname, shape=(512, 768), target_shape=None):
    data = np.fromfile(fname).reshape(shape)
    if target_shape is not None:
        scale_y = shape[0] // target_shape[0]
        scale_x = shape[1] // target_shape[1]
        data = data[::scale_y, ::scale_x]
    radar = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()

    # Normalize to target mean/var
    orig_mean = radar.mean()
    orig_std = radar.std()
    if orig_std > 0:
        radar = (radar - orig_mean) * (TARGET_STD / orig_std) + TARGET_MEAN
    else:
        radar = radar * 0 + TARGET_MEAN
    return radar

def create_folder(args, fname):
    out_path = os.path.join(args.save_dir)
    folder_of_params = os.path.join(out_path, fname)
    if not os.path.exists(folder_of_params):
        os.makedirs(folder_of_params)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(folder_of_params, img_dir), exist_ok=True)
    return folder_of_params

def collect_input_files(input_path):
    """support single file or folder input"""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        file_list = []
        for root, dirs, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith('.bin'):
                    file_list.append(os.path.join(root, f))
        return sorted(file_list)
    else:
        raise ValueError(f"Invalid input path: {input_path}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_image", type=str, required=True, help="File or folder path")
parser.add_argument('--ldm_config', default="configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition_128x192.yaml", type=str)
parser.add_argument('--diffusion_ckpt', default="models/ldm/radial_ldm128x192_epoch=000096.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/radial_imaging_config.yaml", type=str)

parser.add_argument('--simulation', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='results_radial_radarINV')
parser.add_argument('--use_original_steps', action='store_true')
parser.add_argument('--ddim_steps', default=1000, type=int)
parser.add_argument('--ddim_eta', default=0.1, type=float)
parser.add_argument('--n_samples_per_class', default=1, type=int)
parser.add_argument('--ddim_scale', default=1.0, type=float)

parser.add_argument('--step_size_dynamic', default=None, type=float)
parser.add_argument('--step_size_static', default=None, type=float)
parser.add_argument('--measurement_scale', default=1, type=float)
parser.add_argument('--measurement_step_number', default=15, type=int)
parser.add_argument('--resample_sigma', default=80, type=float)
parser.add_argument('--save_process', action='store_true')
parser.add_argument('--test_var', default=0, type=int)
parser.add_argument('--thresh_points', default=0.001, type=float)
args = parser.parse_args()

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
task_config = load_yaml(args.task_config)
device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
print(f"Device set to {device_str}.")
device = torch.device(device_str)

model, config = get_model(args, device=device)
sampler = DDIMSampler(model)

measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning
print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

constraint_method = None

sample_fn = partial(
    sampler.posterior_sampler,
    measurement_cond_fn=measurement_cond_fn,
    operator_fn=operator.forward,
    constraint_fn=constraint_method,
    S=args.ddim_steps,
    cond_method=task_config['conditioning']['main_sampler'],
    conditioning=None,
    ddim_use_original_steps=args.use_original_steps,
    batch_size=args.n_samples_per_class,
    shape=[4, 32, 48],
    verbose=False,
    unconditional_guidance_scale=args.ddim_scale,
    unconditional_conditioning=None,
    eta=args.ddim_eta,
    **vars(args)
)

# Exception) In case of inpainting, generate mask
if measure_config['operator']['name'] == 'inpainting':
    mask_gen = mask_generator(**measure_config['mask_opt'])

from util.rpl import RadarSignalProcessing
RSP = RadarSignalProcessing('CalibrationTable.npy', method='RA', device='cpu')

RBINS = config['data']['params']['train']['params']['RBINS']
ABINS = config['data']['params']['train']['params']['ABINS']
shape = (RBINS, ABINS)

# ------------------------------------------------------------
# Processing
# ------------------------------------------------------------
input_files = collect_input_files(args.input_image)
print(f"Found {len(input_files)} file(s) to process.")

for file_path in input_files:
    fname = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n*************************** Processing {fname} ***************************")

    try:
        data = load_data(file_path, target_shape=shape)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        continue

    folder_of_params = create_folder(args, fname)
    y_n = operator.forward(data.to(device)) if args.simulation else data.to(device)

    ref_img = torch.randn(y_n.shape).to(device)

    # save label
    gt_name = file_path.replace('radars_ra_interp', 'lidars_mask')
    if os.path.exists(gt_name):
        gt_data = load_data(gt_name, target_shape=shape)
        save_image(gt_data, os.path.join(folder_of_params, 'label', f'{fname}_label.png'), normalize=True)

    # save y_n
    save_image(y_n, os.path.join(folder_of_params, 'input', f'{fname}_input.png'), normalize=True)

    # inference
    samples_ddim, _ = sample_fn(original=ref_img, measurement=y_n, fname=fname, test_var=None, folder_of_params=folder_of_params)
    x_samples_ddim = model.decode_first_stage(samples_ddim.detach())
    print(f'sample max {x_samples_ddim.max()}, min: {x_samples_ddim.min()}, avg: {x_samples_ddim.mean()}')

    save_points_radial(x_samples_ddim, fname, folder_of_params, is_lidar=False, thresh=args.thresh_points)
    save_image(x_samples_ddim.mean(dim=1), os.path.join(folder_of_params, 'recon', f'{fname}_recon.png'), normalize=True)

print("✅ All done.")

# sample single file
# python sample_condition_radial.py --input_image demo_data/radars_ra_interp/000038.bin --step_size_static 0.005 --measurement_scale 1.0 --measurement_step_number 100 --gpu 1 --ddim_steps 50 --thresh_points 0.1
# sample original size
# python sample_condition_radial.py --input_image demo_data/radars_ra_interp/000038.bin --step_size_static 0.005 --measurement_scale 1.0 --measurement_step_number 40 --gpu 1 --ddim_steps 50 --thresh_points 0.1 --save_dir results_512x768 --ldm_config configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml --diffusion_ckpt models/ldm/epoch\=000098.ckpt
# sample folder
# python sample_condition_radial.py --input_image /home/icclab/Documents/yyl/RADIal_Dataset/RADIal/dataset_with_labels/radars_ra_interp --step_size_static 0.005 --measurement_scale 1.0 --measurement_step_number 100 --gpu 1 --ddim_steps 50 --thresh_points 0.1
