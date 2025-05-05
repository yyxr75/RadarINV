# RadarINV: Unsupervised Radar Point Cloud Enhancement via LiDAR Guided Diffusion Priors (Submitted to NIPS 2025)


![image](https://github.com/user-attachments/assets/970a529d-9845-43d9-ad94-8769b64e87ee)


## Abstract
In industrial automation, radar is a critical sensor in machine perception. 
However, its angular resolution is inherently limited by the Rayleigh criterion, which depends on both the radar’s operating wavelength and the effective aperture of its antenna array.
Recent methods have leveraged paired LiDAR–radar data for training to achieve notable point enhancement, but this requirement substantially increases model development cost and complexity, limiting scalability and widespread adoption.
To overcome this, we introduce RadarINV, an unsupervised radar points generation enhancement algorithm that employs a LiDAR-guided diffusion model as a prior without the need for paired training data. 
Specifically, our approach reformulates radar angle estimation recovery as an inverse problem and incorporates prior knowledge through a diffusion model with LiDAR domain knowledge during the solution process.
Experimental results demonstrate that our method attains high fidelity and low noise performance compared to traditional regularization techniques, and, relative to paired training methods, it not only achieves comparable performance but also offers generalization capability.
To our knowledge, this is the first approach that enhances radar points output by integrating prior knowledge via a diffusion model rather than relying on paired training data.

## Install 

```
git clone https://github.com/yyxr75/RadarINV-Unsupervised-Radar-Point-Cloud-Enhancement-via-LiDAR-Guided-Diffusion-Priors.git

cd RadarINV-Unsupervised-Radar-Point-Cloud-Enhancement-via-LiDAR-Guided-Diffusion-Priors

```

## Download pretrained model

```
mkdir -p models

wget https://drive.google.com/file/d/1otQjrseEkKl0OgRYx4j5-iI81QZTpnlR/view?usp=drive_link -P ./models
```

## Set envrionment

```
conda env create -f environment.yaml
```

## Inference

```
python sample_condition_radial.py --step_size_dynamic 0.001  --measurement_scale 1.0 --measurement_step_number 20 --unet_lr 0.001 --unet_iters 10 --resample_sigma 80 --save_process --gpu 0
```

## Citation

If you find our work interesting, please consider citing
```
@inproceedings{
title={RadarINV: Unsupervised Radar Point Cloud Enhancement via LiDAR Guided Diffusion Priors},
year={2025},
}
```
