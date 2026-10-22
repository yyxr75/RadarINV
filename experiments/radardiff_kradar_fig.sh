#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; 
_RADARDIFF; PYE=$PY
CK=$SP/radardiff_runs/edm_radial/ema_0.9999_025000.pt
$PYE $SP/prep_kradar_fig.py
cd $RADARDIFF_REPO
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD OMP_NUM_THREADS=4 LD_LIBRARY_PATH=$RADARDIFF_LIB
rm -rf inference_results/kradar_fig_*
$PY scripts/image_sample_radial.py --training_mode edm --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --image_size 128 --num_channels 64 --num_heads 4 --num_res_blocks 3 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --weight_schedule karras --in_ch 2 --out_ch 1 --batch_size 1 --model_path $CK --dataset_dir $SP/radardiff_kradar_fig/ --dataloading_config config/radial.yml --output_dir kradar_fig_ > $SP/radardiff_runs/sample_kradar_fig.log 2>&1
mkdir -p inference_results/kradar_fig_merged; for f in inference_results/kradar_fig_test_1/pre_polar_image/*.png; do cp $f inference_results/kradar_fig_merged/1_$(basename $f); done
$PYE $SP/kradar_radardiff.py topoints inference_results/kradar_fig_merged $SP/radardiff_runs/points_kradar_fig
echo "KRADAR FIG DONE"
