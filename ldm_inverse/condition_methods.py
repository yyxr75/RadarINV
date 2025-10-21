from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import os
from tqdm import tqdm
from util.save_points import save_points_kradar, save_points_radial

__CONDITIONING_METHOD__ = {}
# DEFINE
THRESHOLD = 0.01
USE_L1_OPTIMIZATION = False
OUT_SUBFOLDER_FNAME = f'Ax_L1_Reg' if USE_L1_OPTIMIZATION else f'Ax_L2_Reg'


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, model, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](model=model, operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, model, operator, noiser, **kwargs):
        self.model = model
        self.operator = operator
        self.noiser = noiser
        self.index_prev = 0
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, folder_of_params=None, **kwargs):
        if not measurement.requires_grad:
            measurement.requires_grad = True
            measurement = measurement.to(x_prev.device)
        if self.noiser.__name__ == 'gaussian':
            X_decode = self.model.differentiable_decode_first_stage(x_0_hat) # 
            Ax = self.operator.forward(X_decode, **kwargs) # 
            difference = measurement - Ax
            norm = torch.sum(difference*difference) # L2 norm
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, retain_graph=True)[0]
            
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(self.model.differentiable_decode_first_stage(x_0_hat), **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        else:
            raise NotImplementedError

        return norm_grad, norm


    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, model, operator, noiser, **kwargs):
        super().__init__(model, operator, noiser)
        self.operator = operator

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, scale=None, **kwargs):
        if scale is None:
            scale = 0.01
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        # 新增L2正则化
        # norm_grad = norm_grad+2*x_prev
        # 新增L1正则化
        # norm_grad = norm_grad +torch.sign(x_prev) 
        # proj_step = kwargs.get('proj_step', 1)
        x_t = x_t - norm_grad * scale # * proj_step
        return x_t, norm
    
@register_conditioning_method(name='admm')
class ADMMConditioning(ConditioningMethod):
    def __init__(self, model, operator, noiser, **kwargs):
        super().__init__(model, operator, noiser)
        self.operator = operator
        self.rho = kwargs.get('rho', 1.0)  # ADMM penalty parameter
        self.max_iter = kwargs.get('max_iter', 1)  # Maximum ADMM iterations

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, scale=None, **kwargs):
        # Initialize ADMM variables
        z = x_prev.clone()
        u = torch.zeros_like(x_prev)
        
        for _ in range(self.max_iter):
            # x-update
            x_grad, _ = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_prev = x_prev - scale * x_grad + self.rho * (z - u)
            
            # z-update (proximal operator, can be adjusted based on the specific problem)
            z = torch.clamp(x_prev + u, 0, 1)
            
            # u-update
            u = u + x_prev - z

        return x_prev, torch.linalg.norm(x_prev - z)


# ---------        
# test
# ---------
def save_points(sample, fname=None):
    
    # radar coordinate
    AoA_mat = np.load('/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/resample/CalibrationTable.npy',allow_pickle=True).item()
    azimuth_coord = AoA_mat['Azimuth_table']
    # # 预测
    # out_ra = torch.sigmoid(ret['out'])
    out_ra = torch.sigmoid(1000*(sample-0.01))
    # out_ra_np = out_ra.mean(dim=1).detach().cpu().numpy()
    out_ra_np = out_ra.detach().cpu().numpy()
    out_ra_np = np.squeeze(out_ra_np[0,...])
    # from scipy.ndimage import zoom
    # Apply cubic interpolation to increase the size of out_ra_np by a factor of 4 in both dimensions
    # out_ra_np = zoom(out_ra_np, zoom=(4, 4), order=3)
    threshold = 0.01
    pixel_indices = np.where(out_ra_np > threshold)
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

    # 保存点云
    # path_root = '/scratch/user/yanlongyang/Project1_Diffusion_related/dataset_with_labels/radars_L1_Reg_points'
    # os.makedirs(path_root, exist_ok=True)
    # pcd_filename = os.path.join(path_root, '{:06d}.bin'.format(fname))
    # print(pcd_filename)
    # # if os.path.exists(pcd_filename):
    # #     return
    # if len(real_coords) > 0:
    #     real_coords.tofile(pcd_filename)
    #     print(' Frame {:06d} bin file saved!!!!!'.format(fname))
    return real_coords



if __name__ == '__main__':

    do_inference = True
    # OUT_SUBFOLDER_FNAME = 'L1_Reg'

    from omegaconf import OmegaConf
    import sys
    sys.path.append('/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/resample')
    from data.dataloader import get_dataset, get_dataloader
    from model_loader import load_model_from_config, load_yaml
    import argparse
    from ldm_inverse.measurements import get_noise, get_operator

    def get_model(args):
        config = OmegaConf.load(args.ldm_config)
        model = load_model_from_config(config, args.diffusion_config)
        return model

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--ldm_config', default="configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml", type=str)
    parser.add_argument('--diffusion_config', default="models/ldm/radial_ldm_epoch=000098.ckpt", type=str)
    parser.add_argument('--task_config', default="configs/tasks/radar_imaging_config.yaml", type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--ddim_steps', default=500, type=int)
    parser.add_argument('--ddim_eta', default=0.0, type=float)
    parser.add_argument('--n_samples_per_class', default=1, type=int)
    parser.add_argument('--ddim_scale', default=1.0, type=float)
    
    parser.add_argument('--LR', type=float, default=0.001)

    args = parser.parse_args()


    # Load configurations
    task_config = load_yaml(args.task_config)

    # Device setting
    device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
    print(f"Device set to {device_str}.")
    device = torch.device(device_str)  

    # Loading model
    # model = get_model(args)
    # model.to(device)
    model = None

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
    measurement_cond_grad = cond_method.grad_and_value
    print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

    # Prepare dataloader
    data_config = task_config['data']
    dataset = get_dataset(**data_config)
    dataloader = get_dataloader(dataset, batch_size=args.n_samples_per_class, num_workers=0, train=False)

    # Do inference

    total_iter = 10000
    LR = args.LR

    OUT_SUBFOLDER_FNAME = os.path.join(OUT_SUBFOLDER_FNAME, str(LR))
    save_path = os.path.join(args.save_dir, OUT_SUBFOLDER_FNAME)
    os.makedirs(save_path, exist_ok=True)

    for i, data_dict in enumerate(tqdm(dataloader, desc="Inference Progress")):
        lidar = data_dict['lidar_ra'].to(device)
        y_radar = data_dict['radar_ra'].to(device)
        # if i <3:
            # continue
        fname = data_dict['data_fname'].item()
        if fname != 181:
            print(f'process {fname}')
            continue
        if not do_inference:
            break
        ps = PosteriorSampling(model=model, operator=operator, noiser=noiser)

        B,C,W,H = lidar.shape
        
        # 各种起始条件
        # init_x = torch.randn((B,4,32,48)).to(device).requires_grad_(True)
        # init_x = lidar.requires_grad_(True)
        init_x = torch.randn((B,C,W,H)).to(device).requires_grad_(True)
        
        # 运行优化计算

        for index in range(total_iter):
            grad, loss = measurement_cond_grad(x_prev=init_x, x_0_hat=init_x, measurement=y_radar, index=index, OUT_SUBFOLDER_FNAME=save_path, save_process=True)
            
            if USE_L1_OPTIMIZATION:
                grad_l1 = grad + torch.sign(init_x)
                init_x = init_x - grad_l1 * LR
            else:
                grad_l2 = grad + 2*init_x
                init_x = init_x - grad_l2 * LR

            # if index%10 == 0:
            #     grad_norm = torch.log2(torch.norm(grad_l1 if USE_L1_OPTIMIZATION else grad_l2)).item()
            #     print(f'iter: {index}, loss: {loss.item()}')

        # when latent
        # init_x = model.differentiable_decode_first_stage(init_x)
            if index % 1000 != 0:
                continue
            grad_norm = torch.log2(torch.norm(grad_l1 if USE_L1_OPTIMIZATION else grad_l2)).item()
            print(f'iter: {index}, loss: {loss.item()}')
            plt.figure()
            init_x_4show =  torch.sigmoid(1000 * (init_x.mean(dim=1, keepdim=True) - THRESHOLD))
            init_x_4show =  init_x.mean(dim=1)
            init_x_4show_np = init_x_4show.cpu().detach().numpy().squeeze()
            # np.save(f'results/{OUT_SUBFOLDER_FNAME}/init_x_4show_{i:06d}.npy', init_x_4show_np)
            plt.subplot(1,2,1)
            plt.imshow(init_x_4show_np, cmap='gray')
            # plt.title('L2_result')
            # fname = save_path + f'/result_{fname:06d}.png'

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # X = np.arange(0, init_x_4show.shape[2])
            # Y = np.arange(0, init_x_4show.shape[1])
            # X, Y = np.meshgrid(X, Y)
            # Z = init_x_4show.cpu().detach().numpy().squeeze()
            # ax.plot_surface(X, Y, Z, cmap='viridis')
            # plt.title('3D Mesh of init_x.mean(dim=1)')
            # plt.savefig(f'results/{OUT_SUBFOLDER_FNAME}/3D_mesh_result_{i:06d}.png')
            # plt.close()

            # save points
            ret_points = save_points(init_x_4show, fname)
            plt.subplot(1,2,2, facecolor='black')
            plt.scatter(ret_points[:,0], ret_points[:,1], c='white')
            plt.title('ret_points')
            plt.savefig(f'results/{OUT_SUBFOLDER_FNAME}/result_{fname:06d}_idx_{index:06d}.png', dpi=600)
            plt.close()
            print(f'Frame {fname} saved!!!!')

        break


    '''
    # --------------
    # Assuming lidar is the ground truth image
    # calculate the PSNR, SNR
    # --------------
    init_x_4show = np.load(f'results/{OUT_SUBFOLDER_FNAME}/init_x_4show_{i:06d}.npy')
    init_x_4show = (init_x_4show>0.01)+0.0
    def calculate_psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        PIXEL_MAX = 1.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    def calculate_snr(signal1, signal2):
        signal_power = np.mean(signal1 ** 2)
        noise_power = np.mean((signal1 - signal2) ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * math.log10(signal_power / noise_power)

    ground_truth = lidar.mean(dim=1).cpu().detach().numpy().squeeze()

    generated_image = init_x_4show
    psnr_value = calculate_psnr(ground_truth, generated_image)
    print(f'Generated PSNR value: {psnr_value}')
    snr_value = calculate_snr(ground_truth, generated_image)
    print(f'Generated SNR value: {snr_value}')

    generated_image = y_radar.mean(dim=1).cpu().detach().numpy().squeeze()
    psnr_value = calculate_psnr(ground_truth, generated_image)
    print(f'Radar PSNR value: {psnr_value}')
    snr_value = calculate_snr(ground_truth, generated_image)
    print(f'Radar SNR value: {snr_value}')
    print(f"mean={init_x_4show.mean()}, max={init_x_4show.max()}, min={init_x_4show.min()}")

    radar_show = y_radar.mean(dim=1).cpu().detach().numpy().squeeze()
    plt.imshow(radar_show)
    plt.savefig(f'results/{OUT_SUBFOLDER_FNAME}/radar_out.png')
    plt.close()

    import pdb; pdb.set_trace()
    '''
# python ldm_inverse/condition_methods.py --model_config models/ldm/cin-ldm-vqvae-f8-radial_uncondition.yaml --ldm_config configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml --diffusion_config models/ldm/radial_ldm_epoch=000098.ckpt --task_config configs/tasks/radar_imaging_config.yaml --gpu 0 --save_dir ./results
