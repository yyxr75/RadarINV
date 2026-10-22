#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; 
cd $REPO
export PYTHONPATH=$REPO:$REPO/src/taming-transformers:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export RADARINV_NODEBUG=1 RADARINV_TWEEDIE=1 RADARINV_NEG=1.0 RADARINV_FAST_OP=1 RADARINV_L1=0 RADARINV_CKPT=0
export RADARINV_DEBUG_DIR=/tmp
$PY sample_condition_radial.py \
  --input_image $SP/full/half$1/radars_ra_interp \
  --step_size_static 0.005 --measurement_scale 1.0 --measurement_step_number 40 \
  --gpu $1 --ddim_steps 50 --thresh_points 0.1 \
  --save_dir $SP/full/out$1 \
  --ldm_config configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml \
  --diffusion_ckpt "models/ldm/radial_ldm_epoch=000098.ckpt"
echo "HALF$1_DONE rc=$?"
