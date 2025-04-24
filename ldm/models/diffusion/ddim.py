"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from scripts.utils import *

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr
from scripts.utils import clear_color

import torch.optim as optim
import time
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
def calculate_similarity_metrics(pc1, pc2):
    # Chamfer Distance (CD)
    try:
        d_matrix = pairwise_distances(pc1[:, :2], pc2[:, :2])
    except:
        import pdb;pdb.set_trace()
    cd = np.mean(np.min(d_matrix, axis=0)) + np.mean(np.min(d_matrix, axis=1))
    
    return cd

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        if ddim_num_steps < 1000:
          ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                    ddim_timesteps=self.ddim_timesteps,
                                                                                    eta=ddim_eta,verbose=verbose)
          self.register_buffer('ddim_sigmas', ddim_sigmas)
          self.register_buffer('ddim_alphas', ddim_alphas)
          self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
          self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
              (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                          1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        print('ddim_eta: ', ddim_eta)

        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        Sampling wrapper function for UNCONDITIONAL sampling.
        """

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates


    def posterior_sampler(self, original, measurement, measurement_cond_fn, operator_fn,
               S,
               batch_size,
               shape,
               constraint_fn=None,
               cond_method=None,
               conditioning=None,
               ddim_use_original_steps=False,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               timesteps=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               folder_of_params=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        Sampling wrapper function for inverse problem solving.
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        if cond_method is None or cond_method == 'resample':
            samples, intermediates = self.resample_sampling(original, measurement, measurement_cond_fn,  # conditioning,
                                                        size,
                                                        constraint_fn=constraint_fn,
                                                        operator_fn=operator_fn,
                                                        callback=callback,
                                                        img_callback=img_callback,
                                                        timesteps=timesteps,
                                                        quantize_denoised=quantize_x0,
                                                        mask=mask, x0=x0,
                                                        ddim_use_original_steps=ddim_use_original_steps,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        folder_of_params=folder_of_params,
                                                        **kwargs
                                                        )
            
        else:
            raise ValueError(f"Condition method string '{cond_method}' not recognized.")
        
        return samples, intermediates

    def lr_optimizer(self, iters, total_iters, **kwargs):
        # sin learning rate decay
        start_lr = kwargs.get('start_lr', None)
        end_lr = kwargs.get('end_lr', None)
        lr = start_lr + (end_lr - start_lr) * (1 + math.cos(2*math.pi * iters / total_iters))
        return lr

    def resample_sampling(self, original, measurement, measurement_cond_fn, shape, test_var=None, constraint_fn=None, cond=None, operator_fn=None,
                     inter_timesteps=10, x_T=None, ddim_use_original_steps=False,
                     callback=None, timesteps=None, quantize_denoised=False,
                     mask=None, x0=None, img_callback=None, log_every_t=100,
                     temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                     unconditional_guidance_scale=1., unconditional_conditioning=None, folder_of_params=None, **kwargs):
        """
        DDIM-based sampling function for ReSample.

        Arguments:
            measurement:            Measurement vector y in y=Ax+n.
            measurement_cond_fn:    Function to perform DPS. 
            operator_fn:            Operator to perform forward operation A(.)
            inter_timesteps:        Number of timesteps to perform time travelling.

        """

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        
        img = img.requires_grad_() # Require grad for data consistency

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        # Need for measurement consistency
        alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas 
        alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
        betas = self.model.betas


        measurement_scale = kwargs.get('measurement_scale', None)
        measurement_step_number = kwargs.get('measurement_step_number', None)
        measurement = measurement*measurement_scale

        iter_cnt = 0
        # for i, step in tqdm(enumerate(time_range), desc='DDIM Sampling', total=len(time_range)):
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator): 
            # Instantiating parameters
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device, requires_grad=False) # Needed for ReSampling
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device, requires_grad=False) # Needed for ReSampling
            b_t = torch.full((b, 1, 1, 1), betas[index], device=device, requires_grad=False)            

            # # # 存特征图
            # if index % 100 == 0:
            #     iters_count = index
            #     data_fname = kwargs.get('fname', None)
            #     save_path = os.path.join(folder_of_params, 'middle_results_extra_l1norm')
            #     if type(data_fname) is int:
            #         data_fname = str(data_fname)
            #     save_path = os.path.join(save_path, f'{data_fname}')
            #     os.makedirs(save_path, exist_ok=True)
            #     save_fname = f'{save_path}/iters_{iters_count:06d}.npy'
            #     save_img = img.detach().cpu().numpy()
            #     np.save(save_fname, save_img)
            #     decoded_img = self.model.decode_first_stage(img)
            #     decoded_img = decoded_img.mean(dim=1).cpu().detach().numpy().squeeze()
            #     decoded_img = (decoded_img-decoded_img.min())/(decoded_img.max()-decoded_img.min())
            #     plt.imshow(decoded_img)
            #     plt.colorbar()
            #     plt.axis('off')
            #     plt.savefig(f'{save_path}/iters_{iters_count:06d}_decoded.png', dpi=100)
            #     plt.close()
            #     save_fname = f'{save_path}/iters_{iters_count:06d}_decoded.npy'
            #     np.save(save_fname, decoded_img)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            # Unconditional sampling step
            # pred_x0 is from DDIM, pseudo_x0 is computing \hat{x}_0 using Tweedie's formula
            out, pred_x0, pseudo_x0, e_t = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            # img = out

            step_size_dynamic = kwargs.get('step_size_dynamic', None)
            step_size_static = kwargs.get('step_size_static', None)
            if step_size_dynamic is not None and step_size_static is None:
                step_size = a_t*step_size_dynamic
            elif step_size_static is not None and step_size_dynamic is None:
                step_size = step_size_static
            else:
                raise ValueError('step_size_dynamic and step_size_static must be provided together')
            
            for j in range(measurement_step_number):
                # step_size = a_t*self.lr_optimizer(iters=iter_cnt, total_iters=total_steps*measurement_step_number, **kwargs)
                # iter_cnt += 1
                img, _ = measurement_cond_fn(x_t=out, # x_t is x_{t-1}
                                    measurement=measurement,
                                    # noisy_measurement=measurement,
                                    x_prev=img, # x_prev is x_t, pure noise
                                    # x_ref=original, # 试试用unet的结果当输出
                                    x_0_hat=pseudo_x0, # Tweedie's formula output x_0_hat
                                    # scale=a_t*0.001, # For DPS learning rate / scale, 文章里2.5 medical image, 0.5 natural image
                                    # scale=0.00001, # For DPS learning rate / scale, 文章里2.5 medical image, 0.5 natural image
                                    scale = step_size,
                                    index=index-j,
                                    folder_of_params=folder_of_params,
                                    **kwargs,
                                    )
                out = img
                # twidie formula
                sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if ddim_use_original_steps else self.ddim_sqrt_one_minus_alphas
                sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
                pseudo_x0 = (img - sqrt_one_minus_at**2 * e_t) / a_t.sqrt()
            
            '''

            
            # 构建latent_radar与img的L2损失
            # img = self.constraint_optimization(out.detach(), latent_radar, max_iters=1)
            
            # ----------------exp1-20240830----------------
            if index is not None and index%100==0:
                init_x_for_save = self.model.decode_first_stage(img)
                init_x_for_save = init_x_for_save.mean(dim=1)
                ret_radar_points = Visualizer.save_points_cfar(RSP, init_x_for_save)

                CD_loss = calculate_similarity_metrics(ret_radar_points, ret_lidar_points)
                CD_LOSSES.append(CD_loss.item())
                print(f'iter: {index}, CD_loss: {CD_loss.item()}')

                plt.figure(tight_layout=True, figsize=(10,10), facecolor='black')
                plt.subplot(2,2,1, facecolor='black')
                plt.scatter(ret_lidar_points[:, 0], ret_lidar_points[:, 1], c='white', marker='o', label='Lidar', s=0.1)
                plt.scatter(ret_radar_points[:, 0], ret_radar_points[:, 1], c='red', marker='o', label='Recon', s=0.2)

                plt.legend()
                plt.subplot(2,2,2)
                plt.imshow(measurement.mean(dim=1).cpu().detach().numpy().squeeze(), cmap='gray')
                plt.axis('off')
                plt.subplot(2,2,3)
                # init_x_for_save = torch.sigmoid(1000*(init_x_for_save-0.01))
                plt.imshow(init_x_for_save.cpu().detach().numpy().squeeze(), cmap='jet')
                plt.colorbar()
                plt.axis('off')
                plt.subplot(2,2,4)
                plt.imshow(original.mean(dim=1).cpu().detach().numpy().squeeze(), cmap='gray')
                plt.axis('off')

                OUT_SUBFOLDER_FNAME = kwargs.get('OUT_SUBFOLDER_FNAME', None)
                os.makedirs(OUT_SUBFOLDER_FNAME, exist_ok=True)
                # test_var = kwargs.get('test_var', None)
                plt.savefig(f'{OUT_SUBFOLDER_FNAME}/lidar_radar_comparison_{fname:06d}_test_var_{test_var:06d}_{index:06}.png', dpi=100)
                plt.close()

                np.savetxt(f'{OUT_SUBFOLDER_FNAME}/recon_coords_{fname:06d}_test_var_{test_var:06d}_{index:06}.csv', ret_radar_points, delimiter=',')
                np.savetxt(f'{OUT_SUBFOLDER_FNAME}/lidar_coords_{fname:06d}_test_var_{test_var:06d}_{index:06}.csv', ret_lidar_points, delimiter=',')



            # run radarHD for constraint
            optim_iters = kwargs.get('unet_iters', None)
            optim_lr = kwargs.get('unet_lr', None)
            resample_sigma = kwargs.get('resample_sigma', None)
            if (i+1) % 100 == 0:
                # x_t = img.detach().clone()
                x_t = img
                opt_var = self.constraint_optimization(pseudo_x0.detach(), original, constraint_fn, optim_iters, optim_lr,)
                if index >= 0:
                    sigma = resample_sigma*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)  
                else:
                    sigma = 0.5
                img = self.stochastic_resample(pseudo_x0=opt_var, x_t=x_t, a_t=a_prev, sigma=sigma)
                img = img.requires_grad_() # Seems to need to require grad here
                print('In {}th step, code is running constraint_optimization and stochastic_resample'.format(index))

            # --------------------------------------
            # Instantiating time-travel parameters
            splits = 1 # TODO: make this not hard-coded
            index_split = total_steps // splits

            # Performing time-travel if in selected indices
            # 从2/3*500=334次iter开始，进行resample和pixel/latent优化
            if index <= (total_steps - index_split) and index > 0:   
                x_t = img.detach().clone()

                # Performing only every 10 steps (or so)
                # TODO: also make this not hard-coded
                if index % 1 == 0 :  
                    # 重复跑5次ddim，在扩散模型中，重复运行 p_sample_ddim 的目的是为了在特定的时间步上进行多次迭代，以逐步逼近测量数据 (GPT)
                    for k in range(i, min(i+inter_timesteps, len(list( reversed() ))-1)):
                        step_ = list( reversed(timesteps))[k+1]
                        ts_ = torch.full((b,), step_, device=device, dtype=torch.long)
                        index_ = total_steps - k - 1

                        # Obtain x_{t-k}
                        img, pred_x0, pseudo_x0 = self.p_sample_ddim(img, cond, ts_, index=index_, use_original_steps=ddim_use_original_steps,
                                            quantize_denoised=quantize_denoised, temperature=temperature,
                                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning)
                        print('In {}th step, repeat running p_sample_ddim {}'.format(index, k))
                    # Some arbitrary scheduling for sigma，
                    # sigma是为了在重采样中，权衡x_t和x_0_hat的比例用来获得新的x_t的值。sigma越大，越偏向measurement
                    if index >= 0:
                        sigma = 80*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)  
                    else:
                        sigma = 0.5

                    # Pixel-based optimization for second stage
                    # 在330到166次iter之间，进行pixel优化
                    if index >= index_split: 
                        # Enforcing consistency via pixel-based optimization
                        pseudo_x0 = pseudo_x0.detach() 
                        pseudo_x0_pixel = self.model.decode_first_stage(pseudo_x0) # Get \hat{x}_0 into pixel space

                        opt_var = self.pixel_optimization(measurement=measurement, 
                                                          x_prime=pseudo_x0_pixel,
                                                          operator_fn=operator_fn)
                        # radial sampling
                        opt_var,_,_ = self.model.encode_first_stage(opt_var) # Going back into latent space
                        # rgb sampling
                        # opt_var = self.model.encode_first_stage(opt_var) # Going back into latent space
                        img = self.stochastic_resample(pseudo_x0=opt_var, x_t=x_t, a_t=a_prev, sigma=sigma)
                        img = img.requires_grad_() # Seems to need to require grad here
                        print('In {}th step, code is running pixel_optimization and stochastic_resample'.format(index))

                    # Latent-based optimization for third stage
                    # 在166到0iter之间，进行latent优化
                    elif index < index_split: # Needs to (possibly) be tuned

                        # Enforcing consistency via latent space optimization
                        pseudo_x0, _ = self.latent_optimization(measurement=measurement,
                                                             z_init=pseudo_x0.detach(),
                                                             operator_fn=operator_fn)


                        sigma = 80 * (1-a_prev)/(1 - a_t) * (1 - a_t / a_prev) # Change the 40 value for each task

                        img = self.stochastic_resample(pseudo_x0=pseudo_x0, x_t=x_t, a_t=a_prev, sigma=sigma) 
                        print('In {}th step, code is running latent_optimization and stochastic_resample'.format(index))

               
        # psuedo_x0, _ = self.latent_optimization(measurement=measurement,
        #                                                      z_init=img.detach(),
        #                                                      operator_fn=operator_fn)
        # img = psuedo_x0.detach().clone()
        # print('In {}th final step, code is running latent_optimization'.format(index))
        img = self.constraint_optimization(pseudo_x0.detach(), original, constraint_fn, 200, 0.01,)
        img = img.requires_grad_() 
        print('In {}th step, code is running constraint_optimization and stochastic_resample'.format(index))
        '''
            # Callback functions if needed
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)      

        return img, intermediates

    def pixel_optimization(self, measurement, x_prime, operator_fn, eps=1e-3, max_iters=2000):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            x_prime:               Estimation of \hat{x}_0 using Tweedie's formula
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss() # MSE loss

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=1e-3) # Initializing optimizer
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop

        for _ in range(max_iters):
            optimizer.zero_grad()
            
            # measurement_loss = loss(operator_fn(measurement), operator_fn( opt_var ) ) 
            measurement_loss = loss(measurement, opt_var ) 
            
            measurement_loss.backward() # Take GD step
            optimizer.step()

            # Convergence criteria
            if measurement_loss < eps**2: # needs tuning according to noise level for early stopping
                break

        return opt_var


    def latent_optimization(self, measurement, z_init, operator_fn, eps=1e-3, max_iters=500, lr=None):

        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        
        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()

        if lr is None:
            lr_val = 1e-3
        else:
            lr_val = lr.item()

        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=lr_val) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop
        init_loss = 0
        losses = []
        
        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn( self.model.differentiable_decode_first_stage( z_init ) ))          

            if itr == 0:
                init_loss = output.detach().clone()
                
            output.backward() # Take GD step
            optimizer.step()
            cur_loss = output.detach().cpu().numpy() 
            
            # Convergence criteria

            if itr < 200: # may need tuning for early stopping
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)
                    
            if cur_loss < eps**2:  # needs tuning according to noise level for early stopping
                break


        return z_init, init_loss       

    def constraint_optimization_error(self, pseudo_x0, original, constraint_fn=None, max_iters=100, lr=1e-3, eps=1e-6):
        """
        好像不make sense
        因为这里的constraint_fn是一个unet: y -> x
        但是我这里输入的是\bar{x}_0，不应该是这样输入的。
        那我就直接用original来优化pseudo_x0
        Function to optimize pseudo_x0 based on the constraint function.
        
        Args:
            pseudo_x0 (torch.Tensor): The initial pseudo x0 to be optimized.
            original (torch.Tensor): The original input.
            constraint_fn (function): The constraint function to be applied.
            max_iters (int): Maximum number of optimization iterations.
            lr (float): Learning rate for optimization.
            eps (float): Convergence threshold.

        Returns:
            torch.Tensor: Optimized pseudo_x0.
        """
        if not pseudo_x0.requires_grad:
            pseudo_x0 = pseudo_x0.requires_grad_()

        optimizer = torch.optim.Adam([pseudo_x0], lr=lr)
        
        for i in range(max_iters):
            optimizer.zero_grad()

            # Apply the constraint function
            # pseudo_x0_pixel = self.model.differentiable_decode_first_stage(pseudo_x0)
            loss_dict = constraint_fn.loss(original, pseudo_x0)
            loss = loss_dict['loss']
            
            loss.backward()
            optimizer.step()
            
            if loss.item() < eps:
                break
        
        return pseudo_x0.detach()


    def constraint_optimization(self, pseudo_x0, original, constraint_fn=None, max_iters=100, lr=1e-3, eps=1e-6):
        """
        Function to optimize pseudo_x0 based on the constraint function.
        
        Args:
            pseudo_x0 (torch.Tensor): The initial pseudo x0 to be optimized.
            original (torch.Tensor): The original input.
            constraint_fn (function): The constraint function to be applied.
            max_iters (int): Maximum number of optimization iterations.
            lr (float): Learning rate for optimization.
            eps (float): Convergence threshold.

        Returns:
            torch.Tensor: Optimized pseudo_x0.
        """
        if not pseudo_x0.requires_grad:
            pseudo_x0 = pseudo_x0.requires_grad_()

        optimizer = torch.optim.Adam([pseudo_x0], lr=lr)
        
        for i in range(max_iters):
            optimizer.zero_grad()

            # Apply the constraint function
            # pseudo_x0_pixel = self.model.differentiable_decode_first_stage(pseudo_x0)
            loss = torch.nn.functional.mse_loss(original, pseudo_x0)
            
            loss.backward()
            optimizer.step()
            
            if loss.item() < eps:
                break
        
        return pseudo_x0.detach()


    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        device = self.model.betas.device
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t)/(sigma + 1 - a_t) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t)))


    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        """
        Function for unconditional sampling using DDIM.
        """

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs, pred_x0, _, _ = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas # 这玩意全是0

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        # print('sigma_t: ', sigma_t) # sigma_t 老是0
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # print('a_prev: ', a_prev)
        # print('sigma_t: ', sigma_t)

        # print(x_prev[0, 0, 0, 0].item())
        # print('x_prev[0, 0, 0, 0]: ', x_prev[0, 0, 0, 0].item())
        # print('pred_x0[0, 0, 0, 0]: ', pred_x0[0, 0, 0, 0])
        # print('noise[0, 0, 0, 0]: ', noise[0, 0, 0, 0])

        # Computing \hat{x}_0 via Tweedie's formula
        pseudo_x0 = (x - sqrt_one_minus_at**2 * e_t) / a_t.sqrt()

        return x_prev, pred_x0, pseudo_x0, e_t


    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec



    def ddecode(self, x_latent, cond=None, t_start=50, temp = 1, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps, temperature = temp, 
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec


               
