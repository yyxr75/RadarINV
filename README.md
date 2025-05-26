# Unsupervised Radar Point Cloud Enhancement via Arbitrary LiDAR Guided Diffusion Prior


![image](https://github.com/user-attachments/assets/970a529d-9845-43d9-ad94-8769b64e87ee)


## Abstract
In industrial automation, radar is a critical sensor in machine perception. However, the angular resolution of radar is inherently limited by the Rayleigh criterion, which depends on both the radar’s operating wavelength and the effective aperture of its antenna array. To overcome these hardware-imposed limitations, recent neural network-based methods have leveraged high-resolution LiDAR data, paired with radar measurements, during training to enhance radar point cloud resolution. While effective, these approaches require extensive paired datasets, which are costly to acquire and prone to calibration error. These challenges motivate the need for methods that can improve radar resolution without relying on paired high-resolution ground-truth data. Here, we introduce an unsupervised radar points enhancement algorithm that employs an arbitrary LiDAR-guided diffusion model as a prior without the need for paired training data. Specifically, our approach formulates radar angle estimation recovery as an inverse problem and incorporates prior knowledge through a diffusion model with arbitrary LiDAR domain knowledge. Experimental results demonstrate that our method attains high fidelity and low noise performance compared to traditional regularization techniques. Additionally, compared to paired training methods, it not only achieves comparable performance but also offers improved generalization capability. To our knowledge, this is the first approach that enhances radar points output by integrating prior knowledge via a diffusion model rather than relying on paired training data.

## Install 

```
git clone https://github.com/yyxr75/RadarINV.git

cd RadarINV

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
@misc{yang2025unsupervisedradarpointcloud,
      title={Unsupervised Radar Point Cloud Enhancement via Arbitrary LiDAR Guided Diffusion Prior}, 
      author={Yanlong Yang and Jianan Liu and Guanxiong Luo and Hao Li and Euijoon Ahn and Mostafa Rahimi Azghadi and Tao Huang},
      year={2025},
      eprint={2505.09887},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.09887}, 
}
```
