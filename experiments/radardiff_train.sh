#!/bin/bash
# Radar-Diffusion (Zhang et al., RA-L 2024) EDM stage, retrained on the RADIal train split.
# Same architecture and optimiser settings as the authors' train_edm.sh; microbatch 16 for memory,
# 30K steps with the authors' LR schedule annealed to zero at the end; GPU 1 only.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd $RADARDIFF_REPO
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
$PY_RADARDIFF scripts/edm_train_radial.py --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 32 --microbatch 16 --image_size 128 --lr 0.00005 --num_channels 64 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --in_ch 2 --out_ch 1 --dataset_dir $SP/radardiff_data/ --dataloading_config config/radial.yml --out_dir $SP/radardiff_runs/edm_radial --lr_anneal_steps 30000 --save_interval 5000 --log_interval 100
echo "TRAIN_DONE rc=$?"
