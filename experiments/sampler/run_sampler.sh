#!/bin/bash
# usage: run_sampler.sh <opmode: 1> <zeta> <K> <input_dir> <out_dir>
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; cd $REPO
export PYTHONPATH=$REPO:$REPO/src/taming-transformers:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export RADARINV_NODEBUG=1 RADARINV_TWEEDIE=1 RADARINV_NEG=1.0 RADARINV_FAST_OP=$1 RADARINV_L1=0 RADARINV_CKPT=0 RADARINV_DEBUG_DIR=/tmp
$PY sample_condition_radial.py --input_image $4 --step_size_static $2 --measurement_scale 1.0 --measurement_step_number $3 \
  --gpu 0 --ddim_steps 50 --thresh_points 0.1 --save_dir $5 --ldm_config configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml --diffusion_ckpt "models/ldm/radial_ldm_epoch=000098.ckpt"
