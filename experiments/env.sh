# Paths shared by every script in experiments/. Export any of these before running to override the defaults.
SP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)           # this folder; outputs are written below it
REPO=$(cd "$SP/.." && pwd)                                  # repository root (sampler, models/, configs/)
export RADIAL_ROOT=${RADIAL_ROOT:-/data/RADIal/dataset_with_labels}
export KRADAR_ROOT=${KRADAR_ROOT:-/data/kradar/data}
PY=${PY:-python}                                            # python of the RadarINV environment
PY_RADARDIFF=${PY_RADARDIFF:-python}                        # python of the Radar-Diffusion environment
RADARDIFF_LIB=${RADARDIFF_LIB:-$(dirname "$(dirname "$(command -v "$PY_RADARDIFF")")")/lib}
RADARDIFF_REPO=${RADARDIFF_REPO:-$REPO/third_party/Radar-Diffusion/diffusion_consistency_radar}
NANODIFF_REPO=${NANODIFF_REPO:-$REPO/third_party/nanoDiffusion}   # Diffradar
