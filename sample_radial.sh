export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
python sample_condition_radial.py \
  --input_image demo_data/radars_ra_interp/000038.bin \
  --step_size_static 0.005 \
  --measurement_scale 1.0 \
  --measurement_step_number 40 \
  --gpu 1 \
  --ddim_steps 50 \
  --thresh_points 0.1 \
  --save_dir results_512x768 \
  --ldm_config configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml \
  --diffusion_ckpt models/ldm/epoch=000098.ckpt
