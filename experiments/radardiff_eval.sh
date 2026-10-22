#!/bin/bash
# usage: radardiff_eval.sh <checkpoint.pt> <val|test> <tag>   -> samples with the authors' EDM sampler (40 Heun steps),
# converts to RADIal-frame points and scores them with the paper's protocol.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
CK=$1; SPLIT=$2; TAG=$3
_RADARDIFF
cd $RADARDIFF_REPO
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD OMP_NUM_THREADS=4 LD_LIBRARY_PATH=$RADARDIFF_LIB
CFG=config/radial.yml; [ -f config/radial_${SPLIT}.yml ] && CFG=config/radial_${SPLIT}.yml
rm -rf inference_results/radial_${TAG}_${SPLIT}
$PY scripts/image_sample_radial.py --training_mode edm --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --image_size 128 --num_channels 64 --num_heads 4 --num_res_blocks 3 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --weight_schedule karras --in_ch 2 --out_ch 1 --batch_size 1 --model_path $CK --dataset_dir $SP/radardiff_data/ --dataloading_config $CFG --output_dir radial_${TAG}_ > $SP/radardiff_runs/sample_${TAG}_${SPLIT}.log 2>&1
echo "sampling rc=$?  frames: $(ls inference_results/radial_${TAG}_${SPLIT}/pre_polar_image/*.png 2>/dev/null | wc -l)"
$PY $SP/radardiff_to_points.py inference_results/radial_${TAG}_${SPLIT}/pre_polar_image $SP/radardiff_runs/points_${TAG}_${SPLIT}
$PY $SP/eval_points.py $SP/radardiff_runs/points_${TAG}_${SPLIT} - "RadarDiffusion_${TAG}_${SPLIT}" $SP/radardiff_runs/eval_${TAG}_${SPLIT}.json 2>&1 | grep -v libtinfo
