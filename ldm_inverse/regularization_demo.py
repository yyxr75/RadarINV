from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import os
from tqdm import tqdm
from util.save_points import save_points_kradar, numpy_cfar, save_points_radial


from omegaconf import OmegaConf
import sys
sys.path.append('/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/resample')
from data.dataloader import get_dataset, get_dataloader
from model_loader import load_model_from_config, load_yaml
import argparse
from ldm_inverse.measurements import get_noise, get_operator
# Save CD_LOSSES to a CSV file
import csv
from visualize import visualizer
import torch.nn.functional as F

__CONDITIONING_METHOD__ = {}

# DEFINE
THRESHOLD = 0.01

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
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    # x_prev = ref_img = lidar，测试，为了加噪声让成像更准
    # measurement = radar，测试，为了加噪声让成像更准
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if not measurement.requires_grad:
            measurement.requires_grad = True
            measurement = measurement.cuda() if torch.cuda.is_available() else measurement
        if self.noiser.__name__ == 'gaussian':
            latent_space_test = kwargs.get('latent_space_test', None)
            if latent_space_test:
                X_decode = self.model.differentiable_decode_first_stage( x_0_hat )
            else:
                X_decode = x_prev
            Ax = self.operator.forward(X_decode, **kwargs) 
            # 
            difference = measurement - Ax
            norm = torch.sum(difference*difference) # x^2
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, retain_graph=True)[0]
            # --------visialize------- 
            index = kwargs.get('index', None)
            if index is not None and index % 1000 == 0 and kwargs.get('save_process', False):
                X_decode_4show = (X_decode.mean(dim=1)-X_decode.min())/(X_decode.max()-X_decode.min())
                Ax_4show = (Ax.mean(dim=1)-Ax.min())/(Ax.max()-Ax.min())
                measurement_4show = (measurement.mean(dim=1)-measurement.min())/(measurement.max()-measurement.min())

                difference_4show = (difference.mean(dim=1)-difference.min())/(difference.max()-difference.min())
                init_z_4show = (x_prev.mean(dim=1)-x_prev.min())/(x_prev.max()-x_prev.min())
                norm_grad_4show = (norm_grad.mean(dim=1)-norm_grad.min())/(norm_grad.max()-norm_grad.min())


                X_decode_4show = torch.sigmoid(1000 * (X_decode.mean(dim=1, keepdim=True) - THRESHOLD))
                # Ax_4show = torch.sigmoid(1000 * (Ax.mean(dim=1, keepdim=True) - THRESHOLD))
                Ax_4show = Ax.mean(dim=1, keepdim=True)
                # measurement_4show = torch.sigmoid(1000 * (measurement.mean(dim=1, keepdim=True) - THRESHOLD))
                measurement_4show = measurement.mean(dim=1, keepdim=True)

                # difference_4show = torch.sigmoid(1000 * (difference.mean(dim=1, keepdim=True) - THRESHOLD))
                difference_4show = difference.mean(dim=1, keepdim=True)
                # init_z_4show = torch.sigmoid(1000 * (x_prev.mean(dim=1, keepdim=True) - THRESHOLD))
                init_z_4show = x_prev.mean(dim=1, keepdim=True)
                # norm_grad_4show = torch.sigmoid(1000 * (norm_grad.mean(dim=1, keepdim=True) - THRESHOLD))
                norm_grad_4show = norm_grad.mean(dim=1, keepdim=True)

                fig, axs = plt.subplots(2, 3, figsize=(15, 5))
                axs[0, 0].imshow(X_decode_4show.cpu().detach().numpy().squeeze().clip(0, 1), cmap='jet')
                axs[0, 0].set_title('X_decode')
                axs[0, 1].imshow(Ax_4show[0].cpu().detach().numpy().squeeze().clip(0, 1), cmap='jet')
                axs[0, 1].set_title('Ax')
                axs[0, 2].imshow(measurement_4show[0].cpu().detach().numpy().squeeze().clip(0, 1), cmap='jet')
                axs[0, 2].set_title('Measurement')
                axs[1, 0].imshow(difference_4show.cpu().detach().numpy().squeeze().clip(0, 1), cmap='jet')
                axs[1, 0].set_title('Difference')
                axs[1, 1].imshow(init_z_4show.cpu().detach().numpy().squeeze().clip(0, 1), cmap='jet')
                axs[1, 1].set_title('Init_z')
                axs[1, 2].imshow(norm_grad_4show.cpu().detach().numpy().squeeze().clip(0, 1), cmap='jet')
                axs[1, 2].set_title('Gradient')
                OUT_SUBFOLDER_FNAME = kwargs.get('OUT_SUBFOLDER_FNAME', None)
                process_folder = os.path.join(OUT_SUBFOLDER_FNAME, f'process')
                os.makedirs(process_folder, exist_ok=True)
                plt.savefig(f'{process_folder}/Ax_parallel_{index}_loss_{norm.item():.4f}.png')
                plt.close()

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
            scale = 0.3
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * scale
        return x_t, norm

def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)

    return model

# Convert range-angle coordinates to Cartesian coordinates
def ra_to_xy(coords, range_reso, angle_reso):
    try:
        ranges = coords[:, 0] * range_reso
        angles = coords[:, 1] * angle_reso * np.pi / 180  # Convert to radians
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
    except:
        return np.zeros((0,2))
    return np.stack((x, y), axis=-1)

    # Calculate Chamfer Distance
def chamfer_distance(x, y):
    x = np.expand_dims(x, axis=1).astype(np.float32)
    y = np.expand_dims(y, axis=0).astype(np.float32)
    dist = np.sum((x - y) ** 2, -1)
    min_dist_xy = np.min(dist, 1)[0]
    min_dist_yx = np.min(dist, 0)[0]
    cd = np.mean(min_dist_xy) + np.mean(min_dist_yx)
    return cd

# from scipy.spatial.distance import directed_hausdorff
# from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
def calculate_similarity_metrics(pc1, pc2):
    # Chamfer Distance (CD)
    try:
        d_matrix = pairwise_distances(pc1[:, :2], pc2[:, :2])
    except:
        import pdb;pdb.set_trace()
    cd = np.mean(np.min(d_matrix, axis=0)) + np.mean(np.min(d_matrix, axis=1))

    return cd #, mhd, ucd, umhd, hd, emd
# ---------        
# test
# ---------
# if __name__ == '__main__':
def main(USE_L1_OPTIMIZATION, OUT_SUBFOLDER_FNAME):
    USE_L1_OPTIMIZATION = False
    def get_model(args):
        config = OmegaConf.load(args.ldm_config)
        model = load_model_from_config(config, args.diffusion_ckpt)
        return model

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    # parser.add_argument('--ldm_config', default="configs/latent-diffusion/cin-ldm-vqvae-f8-coloradar_uncondition.yaml", type=str)
    # parser.add_argument('--diffusion_config', default="models/ldm/coloradar_ldm_epoch=000003.ckpt", type=str)
    # parser.add_argument('--task_config', default="configs/tasks/coloradar_imaging_config.yaml", type=str)
    parser.add_argument('--ldm_config', default="configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml", type=str)
    parser.add_argument('--diffusion_ckpt', default="models/ldm/radial_ldm_epoch=000098.ckpt", type=str)
    # parser.add_argument('--task_config', default="configs/tasks/kradar_imaging_config.yaml", type=str)
    parser.add_argument('--task_config', default="configs/tasks/radial_imaging_config.yaml", type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results_radial_fullscale_norm/L12_Reg')
    # parser.add_argument('--save_dir', type=str, default='./results_kradar/L12_Reg')
    parser.add_argument('--n_samples_per_class', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--total_iter', default=10000, type=int)
    parser.add_argument('--meas_scale', default=1.0, type=float)
    parser.add_argument('--latent_space_test', action='store_true')
    parser.add_argument('--save_process', action='store_true')
    args = parser.parse_args()

    # space_name = 'Latent' if args.latent_space_test else 'Pixel'
    # args.OUT_SUBFOLDER_FNAME = os.path.join(args.save_dir, f'radars_{space_name}_L2_Reg_points_{args.lr}_{args.total_iter}_{args.meas_scale}')
    args.OUT_SUBFOLDER_FNAME =os.path.join(OUT_SUBFOLDER_FNAME, f'measurement_scale_{args.meas_scale}_lr_{args.lr}_total_iter_{args.total_iter}')
    os.makedirs(args.OUT_SUBFOLDER_FNAME, exist_ok=True)
 
    # Load configurations
    task_config = load_yaml(args.task_config)

    # Device setting
    device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
    print(f"Device set to {device_str}.")
    device = torch.device(device_str)  

    # Loading model
    if args.latent_space_test:
        model = get_model(args)
        model.to(device)
    else:
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
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )
    dataset = get_dataset(**data_config) #, transforms=transform)
    dataloader = get_dataloader(dataset, batch_size=args.n_samples_per_class, num_workers=0, train=False)

    ps = PosteriorSampling(model=model, operator=operator, noiser=noiser)

    # Do inference
    do_inference = True
    total_iter = args.total_iter
    range_reso = 103/512
    angle_reso = 180/768

    test_viriance_iters = 1

    Visualizer = visualizer()
    sys.path.append('/scratch/project/cpautodriving/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning')
    from RADIal.SignalProcessing.rpl import RadarSignalProcessing
    RSP = RadarSignalProcessing('/scratch/project/cpautodriving/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/RADIal/SignalProcessing/CalibrationTable.npy',method='RA',device='cpu') #,lib='PyTorch')

    for k in range(test_viriance_iters):
        print(f'test vairantion {k}')
        
        for i, data_dict in enumerate(dataloader):

            fname = data_dict['data_fname'][0]
            if fname < 37:
                print(f'skip {fname}')
                continue

            lidar = data_dict['lidar_ra'].float().to(device)/data_dict['lidar_ra'].max()
            B,C,W,H = lidar.shape
            x_lidar_interpolated = F.interpolate(lidar.float(), size=task_config['measurement']['operator']['shape'], mode='bilinear', align_corners=False)
            x_lidar = x_lidar_interpolated*args.meas_scale

            # save_points_kradar(lidar, fname, args.OUT_SUBFOLDER_FNAME, is_lidar=True)

            save_points_radial(lidar, fname.item(), args.OUT_SUBFOLDER_FNAME, is_lidar=True)

            #插值, for coloradar dataset
            y_radar_raw = data_dict['radar_ra'].float().to(device) # /data_dict['radar_ra'].max()
            y_radar_interpolated = F.interpolate(y_radar_raw[:,:,:200,:].float(), size=task_config['measurement']['operator']['shape'], mode='bilinear', align_corners=False)
            y_radar = y_radar_interpolated*args.meas_scale
            # plt.imshow(y_radar_interpolated[0,0,:,:].cpu().detach().numpy())
            # plt.colorbar()
            # plt.savefig(f"radar_points_{fname}.png")

            print(f'process {fname}')

            # 各种起始条件
            if args.latent_space_test:
                lidar_latent = model.encode_first_stage(x_lidar)[0]
                init_x = torch.randn(lidar_latent.shape).to(device).requires_grad_(True) # coloradar latent init
            else:
                init_x = torch.randn(x_lidar.shape).to(device).requires_grad_(True) # pixel

            # 运行优化计算
            CD_LOSSES = []
            for index in tqdm(range(total_iter), desc="Processing iterations"):

                grad, loss = measurement_cond_grad(x_prev=init_x, x_0_hat=init_x, measurement=y_radar, index=index, **vars(args))

                if USE_L1_OPTIMIZATION:
                    grad_l1 = grad + torch.sign(init_x)
                    init_x = init_x - grad_l1 * args.lr
                else:
                    grad_l2 = grad + 2*init_x
                    init_x = init_x - grad_l2 * args.lr

            # save the final result
            init_x_cfar, thresh = numpy_cfar(init_x.mean(dim=1).cpu().detach().numpy().squeeze(), guard_cells=6, training_cells=3, false_alarm_rate=1.15)
            # save_points_kradar(init_x, fname, args.OUT_SUBFOLDER_FNAME)
            save_points_radial(init_x_cfar, fname.item(), args.OUT_SUBFOLDER_FNAME)
            # plt.figure(figsize=(10, 10), facecolor='black', dpi=100)
            # # plt.scatter(radar_points[:, 0], radar_points[:, 1], s=0.1, c='white')
            # plt.subplot(121)
            # plt.imshow(init_x_cfar, cmap='jet')
            # plt.subplot(122)
            # plt.imshow(thresh, cmap='jet')
            # plt.savefig(os.path.join(args.OUT_SUBFOLDER_FNAME, f"radar_points_{fname}.png"), bbox_inches='tight', pad_inches=0)
            # plt.close()
            break


if __name__ == '__main__':

    LR = 0.001
    # 运行L1优化
    # data_root = './results_kradar_L1_L2'
    data_root = './results_radial_fullscale_norm_L1_L2'
    import threading

    def run_l1_optimization():
        save_dir = os.path.join(data_root, 'L1_Reg_test1')
        os.makedirs(save_dir, exist_ok=True)
        main(USE_L1_OPTIMIZATION=True, OUT_SUBFOLDER_FNAME=save_dir)

    def run_l2_optimization():
        save_dir = os.path.join(data_root, 'L2_Reg_test1')
        os.makedirs(save_dir, exist_ok=True)
        main(USE_L1_OPTIMIZATION=False, OUT_SUBFOLDER_FNAME=save_dir)

    thread1 = threading.Thread(target=run_l1_optimization)
    # thread2 = threading.Thread(target=run_l2_optimization)

    thread1.start()
    # thread2.start()

    thread1.join()
    # thread2.join()

# export PYTHONPATH=$PYTHONPATH:$(pwd); python ldm_inverse/regularization_demo.py --save_process # --latent_space_test