#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; K=$SP/kr; cd $REPO
export PYTHONPATH=$REPO:$REPO/src/taming-transformers:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export RADARINV_NODEBUG=1 RADARINV_TWEEDIE=1 RADARINV_NEG=1.0 RADARINV_FAST_OP=1 RADARINV_L1=0 RADARINV_CKPT=0 RADARINV_DEBUG_DIR=/tmp RADARINV_AZSPAN=53
$PY sample_condition_radial.py --input_image $K/ours_in/radars_ra_interp --task_config configs/tasks/kradar_ours_eval.yaml \
  --step_size_static 0.0005 --measurement_scale 1.0 --measurement_step_number 5 --gpu 0 --ddim_steps 500 --thresh_points 0.1 --save_dir $K/ours_final \
  --ldm_config configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml --diffusion_ckpt "models/ldm/radial_ldm_epoch=000098.ckpt" > $K/ours_final.log 2>&1
echo "rc=$?"; echo "KR FINAL DONE"
