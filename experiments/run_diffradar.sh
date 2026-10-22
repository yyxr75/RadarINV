#!/bin/bash
# Diffradar baseline on the official RADIal splits, GPU 0 only.
# 1) three candidate checkpoints on the first 20 unique val frames  2) pick by val CD  3) 424 test frames with the winner
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; 
ND=$NANODIFF_REPO

export O=$SP/diffradar; mkdir -p $O
export PYTHONPATH=$ND CUDA_VISIBLE_DEVICES=0
cd $ND
C1=checkpoints/ddpm_ddpm-2025-11-21-09-32/Condition_DDPM-RADIal_DDPM_ddpm-2025-11-21-09-32-iteration-98000-model_states.pkl
C0=checkpoints/ddpm_ddpm-2025-03-30-01-58/Condition_DDPM-RADIal_DDPM_ddpm-2025-03-30-01-58-iteration-194000-model_states.pkl
run(){ tag=$1; shift; echo "== $(date '+%F %T') start $tag"; $PY sample_split.py --save_dir $O/$tag "$@" 2>&1 | grep -E "^\[|DONE|loaded|Traceback|Error" ; $PY $SP/eval_points.py $O/$tag/radar_points - "$tag" $O/$tag/eval.json; }
run val20_nov21_raw   --model_path $C1 --split val --limit 20
run val20_nov21_ema   --model_path $C1 --split val --limit 20 --use_ema
run val20_mar30_T1000 --model_path $C0 --split val --limit 20 --num_timesteps 1000 --stride 4
best=$($PY - <<'PY2'
import json,os
O=os.environ["O"]
c={t:json.load(open(f"{O}/{t}/eval.json"))["summary"]["CD"] for t in ["val20_nov21_raw","val20_nov21_ema","val20_mar30_T1000"] if os.path.exists(f"{O}/{t}/eval.json")}
print(min(c,key=c.get))
PY2
)
echo "== winner on val20: $best"; echo "$best" > $O/WINNER
case $best in
  val20_nov21_raw)   run test_diffradar --model_path $C1 --split test ;;
  val20_nov21_ema)   run test_diffradar --model_path $C1 --split test --use_ema ;;
  val20_mar30_T1000) run test_diffradar --model_path $C0 --split test --num_timesteps 1000 --stride 4 ;;
esac
echo "== $(date '+%F %T') ALL DONE"
