#!/bin/bash

measurement_scale_list=(2.5 2.0 1.5 1.0 0.5)
lr_list=(0.0001 0.0005 0.001 0.005 0.01)

for measurement_scale in "${measurement_scale_list[@]}"; do
    for lr in "${lr_list[@]}"; do
        export PYTHONPATH=$PYTHONPATH:$(pwd); python ldm_inverse/regularization_demo.py --save_process --meas_scale "$measurement_scale" --lr "$lr"
    done
done