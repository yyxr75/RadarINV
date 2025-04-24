"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddpm import Layout2ImgDiffusion
import matplotlib.pyplot as plt

sys.path.append('/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning')
# sys.path.append('/root/autodl-tmp/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning')
from RADIal.SignalProcessing.rpl import RadarSignalProcessing
RSP = RadarSignalProcessing('/scratch/user/yanlongyang/Project1_Diffusion_related/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/RADIal/SignalProcessing/CalibrationTable.npy',method='RA',device='cpu') #,lib='PyTorch')
# RSP = RadarSignalProcessing('/root/autodl-tmp/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/RADIal/SignalProcessing/CalibrationTable.npy',method='RA',device='cpu') #,lib='PyTorch')
from visualize import visualizer

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

RBINS = 512
ABINS = 768
def get_radar(input_filename):
    # a = Image.open(input_filename)
    parts = input_filename.split('/')
    parts[-2] = 'radars_ra_interp'
    input_filename = '/'.join(parts)
    a = np.fromfile(input_filename)
    a = np.interp(np.linspace(0, len(a)-1, RBINS*ABINS), np.arange(len(a)), a)
    a = a.reshape(1, RBINS, ABINS)
    ra = (a-a.min())/(a.max()-a.min())
    X = torch.Tensor(ra)
    return X


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--visualFlag",
        type=str,
        nargs="?",
        help="wether visualize ldm or autoencoder",
        default="ldm"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--ddim",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()

    # load 模型
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # load 数据
    data = instantiate_from_config(config.data)
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # 保存结果
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    Visualizer = visualizer()
    visualize_flag = opt.visualFlag = opt.visualFlag

    batch_size = opt.n_samples


    def savebin(path, data):
        with open(path, 'wb') as f:
            f.write(data)

    def log_local(save_dir, log, indexes):
        root = os.path.join(save_dir, "samplesx4")
        os.makedirs(root, exist_ok=True)
        cnt = 0
        for index in indexes:
            try:
                # Create a figure and a set of subplots
                if visualize_flag == 'ldm':
                    ordered_keys = ['inputs', 'reconstruction', 'samples', 'latent_input', 'latent_samples', 'latent_conditioning', 'conditioning']
                elif visualize_flag == 'autoencoder':
                    ordered_keys = ['inputs', 'reconstruction', 'latent_input']
                fig, axes = plt.subplots(1, len(ordered_keys), figsize=(15, 5))
                for ax, key in zip(axes, ordered_keys):
                    try:
                        if key in log:
                            value = log[key]
                            images = value[cnt].cpu().numpy() if value.device.type == 'cuda' else value[cnt].numpy()
                            images = (images - images.min()) / (images.max() - images.min())  # Rescale images
                            images = np.squeeze(images)  # Squeeze images
                            
                            grid = make_grid(torch.from_numpy(images), nrow=3)  # Create a grid of images
                            grid = grid.permute(1, 2, 0)  # Permute to correct the dimension for displaying
                            grid = (grid * 255).type(torch.uint8)  # Convert to uint8 for saving as image
                            
                            ax.imshow(grid.numpy())
                            ax.set_title(key)
                            ax.axis('off')
                    except:
                        import pdb;pdb.set_trace()
                
                plt.tight_layout()
                filename = f'{index:06d}.png'
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                plt.savefig(path)  # Save the figure
                plt.close()  # Close the plot to free memory
                cnt += 1
            except:
                import pdb;pdb.set_trace()
            
            # Save bin for further use
            # Visualizer.save_points(RSP, sample_img, index)

    cnt = 0
    total_cnt = 0
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            tic = time.time()
            for dataloader in [data.train_dataloader(), data.val_dataloader()]:
                for batch_dict in dataloader:
                    radar_size = len(batch_dict['radar_ra'])
                    total_cnt += radar_size
                    print(total_cnt)
                    indexes = batch_dict['data_fname']
                    if visualize_flag == 'ldm' or visualize_flag == 'dps':
                        with model.ema_scope():
                            log = model.log_images(
                            batch_dict, 
                            N=radar_size, 
                            dps=visualize_flag == 'dps',
                            quantize_denoised=False, 
                            inpaint=False, 
                            plot_denoise_rows=False, 
                            plot_progressive_rows=False, 
                            plot_diffusion_rows=False
                        )
                    elif visualize_flag == 'autoencoder':
                        log = model.log_images(
                            batch_dict, 
                            # N=radar_size, 
                            # quantize_denoised=False, 
                            # inpaint=False, 
                            # plot_denoise_rows=False, 
                            # plot_progressive_rows=False, 
                            # plot_diffusion_rows=False
                        )
                    else:
                        raise ValueError(f"Invalid visualize flag: {visualize_flag}")
                    log_local(outpath, log, indexes)
                    import pdb;pdb.set_trace()

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()

# python scripts/unconditioning_ldm_radial_sampling.py --ckpt logs/2024-06-25T23-08-41_vqvae_32x48x8_radial/checkpoints/last.ckpt --config logs/2024-06-25T23-08-41_vqvae_32x48x8_radial/configs/2024-06-27T21-39-30-project.yaml --outdir outputs/autoencoder/ --visualFlag autoencoder