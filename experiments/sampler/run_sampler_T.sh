#!/bin/bash
# usage: run_sampler_T.sh <zeta> <K> <T> <input_dir> <out_dir>   (paper operator path, real radar)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; cd $REPO
export PYTHONPATH=$REPO:$REPO/src/taming-transformers:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export RADARINV_NODEBUG=1 RADARINV_TWEEDIE=1 RADARINV_NEG=1.0 RADARINV_FAST_OP=1 RADARINV_L1=0 RADARINV_CKPT=0 RADARINV_DEBUG_DIR=/tmp
$PY sample_condition_radial.py --input_image $4 --step_size_static $1 --measurement_scale 1.0 --measurement_step_number $2 \
  --gpu 0 --ddim_steps $3 --thresh_points 0.1 --save_dir $5 --ldm_config configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml --diffusion_ckpt "${DIFF_CKPT:-models/ldm/radial_ldm_epoch=000098.ckpt}"
