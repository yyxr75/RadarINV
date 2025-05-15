from ldm_inverse.condition_methods import get_conditioning_method
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
import torch.nn.functional as F
from util.save_points import save_points_kradar, save_points_radial, polar_to_cartesian


def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_ckpt)

    return model

def create_folder(args, fname, index=None):
    out_path = os.path.join(args.save_dir)
    params = 'test_file_number_{}_step_size_dynamic_{}_step_size_static_{}_measurement_scale_{}_measurement_step_number_{}_ldm_timesteps_{}_fname_{}_index_{}'.format(args.test_filenumber, args.step_size_dynamic, args.step_size_static, args.measurement_scale, args.measurement_step_number, args.ldm_timesteps, fname, index)
    folder_of_params = os.path.join(out_path, params)
    if not os.path.exists(folder_of_params):
        os.makedirs(folder_of_params)
    else:
        print(f"Folder {folder_of_params} already exists.")
        exit()

    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(folder_of_params, img_dir), exist_ok=True)

    return folder_of_params

def main(index=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    # use kradar prior

    # use radial prior
    parser.add_argument('--ldm_config', default="configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml", type=str)
    parser.add_argument('--diffusion_ckpt', default="models/epoch=000323.ckpt", type=str)
    parser.add_argument('--task_config', default="configs/tasks/kradar_imaging_config.yaml", type=str)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results_kradar_lr_decay')
    parser.add_argument('--ddim_steps', default=500, type=int)
    parser.add_argument('--ddim_eta', default=0.0, type=float)
    parser.add_argument('--n_samples_per_class', default=1, type=int)
    parser.add_argument('--ddim_scale', default=1.0, type=float)

    # scan parameters
    parser.add_argument('--test_filenumber', default=5, type=int)
    parser.add_argument('--step_size_dynamic', default=0.0005, type=float)
    parser.add_argument('--step_size_static', default=None, type=float)
    parser.add_argument('--measurement_scale', default=1.0, type=float)
    parser.add_argument('--measurement_step_number', default=5, type=int)
    parser.add_argument('--ldm_timesteps', default=1000, type=int)

    parser.add_argument('--start_lr', default=0.001, type=float)
    parser.add_argument('--end_lr', default=1e-8, type=float)
    args = parser.parse_args()

    # ------------------------------------------------------------
    # initialize
    # ------------------------------------------------------------
    # Load configurations
    task_config = load_yaml(args.task_config)
    config = OmegaConf.load(args.ldm_config)
    latent_shape = [int(config['model']['params']['channels']), int(config['model']['params']['image_size'][0]), int(config['model']['params']['image_size'][1])]
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
    # Instantiating sampler
    sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                            S=args.ddim_steps,
                                            cond_method=task_config['conditioning']['main_sampler'],
                                            conditioning=None,
                                            timesteps=args.ldm_timesteps,
                                            ddim_use_original_steps=False,
                                            batch_size=args.n_samples_per_class,
                                            # shape=[4, 16, 4], # Dimension of latent space for radarHD
                                            shape=latent_shape, # Dimension of latent space for coloradar
                                            # shape=[3, 64, 64], # original code
                                            verbose=False,
                                            unconditional_guidance_scale=args.ddim_scale,
                                            unconditional_conditioning=None, 
                                            eta=args.ddim_eta,
                                            **vars(args))

    # Prepare dataloader
    data_root = task_config['data']['root']
    input_files_list = []
    for file in os.listdir(data_root):
        fname = os.path.join(data_root, file)
        if file.split('_')[0] == 'radar':
            input_files_list.append(fname)
    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(**measure_config['mask_opt'])

    # ------------------------------------------------------------
    # Do inference
    # ------------------------------------------------------------
    test_filenumber = args.test_filenumber
    import random

    # Set a seed for reproducibility
    random.seed(42)

    for idx, file in enumerate(input_files_list):
        try:
            data = np.fromfile(file).reshape(task_config['data']['RBINS'], task_config['data']['ABINS'])
        except:
            import pdb;pdb.set_trace()
            pass
        radar = torch.from_numpy(data).squeeze().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 512, 768)
        fname = file.split('/')[-1].split('.')[0] # str

        folder_of_params = create_folder(args, fname, index=idx)

        print('***************************go with {}   ***************************'.format(fname))

        y_n = radar
        y_n_interpolated = F.interpolate(y_n.float(), size=task_config['measurement']['operator']['shape'], mode='bilinear', align_corners=False)
        y_n = y_n_interpolated

        # save input
        input_img = polar_to_cartesian(ref_img)
        plt.figure()
        plt.imshow(input_img)
        plt.axis('off')
        plt.savefig(os.path.join(folder_of_params, 'input', str(fname)+'_input.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        print('saved {}'.format(fname))
        # save label
        label_img = polar_to_cartesian(y_n)
        plt.figure()
        plt.imshow(label_img)
        plt.axis('off')
        plt.savefig(os.path.join(folder_of_params, 'label', str(fname)+'_label.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        print('saved {}'.format(fname))
        # Sampling
        samples_ddim, _ = sample_fn(original=ref_img, measurement=y_n, fname=fname, test_var=None, folder_of_params=folder_of_params) #, constraint_fn=None, conditioning=cond)

        x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

        # save points
        # save_points_radial(x_samples_ddim, fname, folder_of_params)
        save_points_kradar(x_samples_ddim, fname, folder_of_params)


        # save recon
        # output_img = torch.sigmoid(1000*(x_samples_ddim-0.01))
        output_img = polar_to_cartesian(x_samples_ddim)
        plt.figure()
        plt.imshow(output_img, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(folder_of_params, 'recon', str(fname)+'_recon.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        print('saved {}'.format(fname))

if __name__ == '__main__':
    main()

# python sample_condition_radarHD.py --test_filenumber 5 --step_size_dynamic 0.001 --measurement_scale 1.0 --measurement_step_number 20


