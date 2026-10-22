#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; 
export RADARINV_LIK=laplace; $SP/sampler/run_sampler.sh 1 0.005 40 $SP/fair/test53/radars_ra_interp $SP/fair/ours_laplace53 > $SP/fair/laplace.log 2>&1; echo "laplace rc=$?"; echo "LAPLACE DONE"
