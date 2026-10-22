# RadarINV: Unsupervised Radar Range-Azimuth Super-Resolution via LiDAR Guided Diffusion Prior

![image](https://github.com/user-attachments/assets/970a529d-9845-43d9-ad94-8769b64e87ee)

RadarINV enhances the angular resolution of automotive radar without paired radar–LiDAR training data.
Radar enhancement is cast as a cross-modality Bayesian inverse problem:

- **Prior.** A latent diffusion model (VQ-VAE + LDM) is trained on LiDAR occupancy maps only, on a polar
  bird's-eye-view (BEV) grid aligned with the radar range–azimuth (RA) heatmap.
- **Likelihood.** A differentiable radar forward operator `A(x)` maps an occupancy map to the RA heatmap the
  array would measure (main lobe and sidelobes of the array).
- **Inference.** Diffusion posterior sampling: at every DDIM step the latent is pulled towards the radar
  measurement through `A(x)`. The radar is used only at inference, so one prior serves different radar sensors
  (RADIal and K-Radar) without retraining.

This branch contains the code of the revised paper (IEEE TETC revision): the sampler, the evaluation protocol,
the baselines, and every script behind the tables and figures.

---

## 1. Repository layout

```
.
├── sample_condition_radial.py     # posterior sampling entry point (RADIal and K-Radar)
├── model_loader.py                # builds the LDM + VQ-VAE from config and checkpoint
├── ldm_inverse/
│   ├── measurements.py            # radar forward operator A(x) (RadarImagingOperator)
│   └── condition_methods.py       # likelihood gradient (Gaussian / Laplace), non-negativity
├── ldm/                           # latent diffusion (DDIM sampler with Tweedie-guided updates)
├── src/taming-transformers/       # VQ-VAE (taming), with optional activation checkpointing
├── configs/
│   ├── latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml   # prior used in the paper (512x768)
│   └── tasks/{radial_imaging_config,kradar_ours_eval}.yaml        # operator settings per dataset
├── CalibrationTable.npy           # RADIal array calibration (steering vectors, azimuth table)
├── experiments/                   # every script behind the paper's tables and figures (section 5)
├── training/                      # LDM training code used for the Countryside / City-held-out priors
├── baselines/radar_diffusion/     # files added to Radar-Diffusion to retrain it on RADIal
├── environment_radarinv.yml       # exact environment used for the paper (torch 2.0.1, PL 1.9.5)
└── environment.yaml               # original (older) environment of the first release
```

## 2. Installation

```bash
git clone https://github.com/yyxr75/RadarINV.git
cd RadarINV
conda env create -f environment_radarinv.yml      # creates env "radarinv"
conda activate radarinv
export PYTHONPATH=$PWD:$PWD/src/taming-transformers:$PYTHONPATH
```

## 3. Data

| Dataset | What is used | Where to set it |
|---|---|---|
| [RADIal](https://github.com/valeoai/RADIal) (`dataset_with_labels`) | `radars_ra_interp/` (radar RA heatmaps, input), `lidars_no_ground/` (ground-removed LiDAR, GT), `lidars_mask/` (LiDAR polar occupancy, prior training), `ImageSets/{train,val,test}.txt`, `radialx_infos_*.pkl` | `export RADIAL_ROOT=/path/to/RADIal/dataset_with_labels` |
| [K-Radar](https://github.com/kaist-avelab/K-Radar) sequences 1, 2, 3, 4, 6, 8 | `radar_rea_cube/`, `lidar_rea_mask/`, `testset.txt`, `valset.txt` | `export KRADAR_ROOT=/path/to/kradar/data` |

Splits are by sequence: 4,533 annotated RADIal frames from 73 sequences, 3,555 / 554 / 424 frames for
train / val / test (68 / 3 / 2 sequences, none shared). `experiments/frames424.txt` lists the 424 test frames.

## 4. Pretrained weights

All weights are in the Google Drive folder **`RadarINV_weights_2026-10-22`** (link: _to be added_).

| File | Size | Place it at | Used for |
|---|---|---|---|
| `vq_radial_epoch323.ckpt` | 863 MB | `models/first_stage_models/epoch=000323.ckpt` | VQ-VAE (first stage) of every prior |
| `ldm_all_epoch098.ckpt` | 16.8 GB | `models/ldm/radial_ldm_epoch=000098.ckpt` | **main prior** (all scenarios): Tables III, IV, VII, VIII, Fig. 4, K-Radar |
| `ldm_countryside_sd.ckpt` | 8.6 GB | `models/ldm/ldm_countryside_sd.ckpt` | Countryside-only prior (Table V, Fig. 6) |
| `ldm_all_cityheldout_sd.ckpt` | 8.6 GB | `models/ldm/ldm_all_cityheldout_sd.ckpt` | All-scenario prior with three City drives held out (Fig. 6) |
| `radardiff_radial_ema_025000.pt` | 136 MB | anywhere; pass to `experiments/radardiff_eval.sh` | Radar-Diffusion baseline retrained on RADIal |

```bash
mkdir -p models/first_stage_models models/ldm
# download the files above from the Drive folder, then e.g.
mv vq_radial_epoch323.ckpt  "models/first_stage_models/epoch=000323.ckpt"
mv ldm_all_epoch098.ckpt    "models/ldm/radial_ldm_epoch=000098.ckpt"
```

The `*_sd.ckpt` files hold only the `state_dict` (no optimizer state); they load the same way.

## 5. Inference (RADIal)

Settings of the paper (Table II): DDIM `T = 50` steps, `K = 40` likelihood steps per DDIM step,
step size `ζ = 0.005`, Tweedie-guided updates, non-negativity, the fast operator.

```bash
export RADIAL_ROOT=/path/to/RADIal/dataset_with_labels
export RADARINV_NODEBUG=1 RADARINV_TWEEDIE=1 RADARINV_NEG=1.0 RADARINV_FAST_OP=1 RADARINV_L1=0 RADARINV_CKPT=0
python sample_condition_radial.py \
  --input_image $RADIAL_ROOT/radars_ra_interp \
  --step_size_static 0.005 --measurement_scale 1.0 --measurement_step_number 40 \
  --ddim_steps 50 --thresh_points 0.1 --gpu 0 \
  --ldm_config configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml \
  --diffusion_ckpt "models/ldm/radial_ldm_epoch=000098.ckpt" \
  --save_dir results_radial
```

`--input_image` takes one `.bin` RA heatmap or a folder of them. For each frame the sampler writes the decoded
occupancy map `<id>/<id>_decoded.npy`. `experiments/sampler/run_sampler_T.sh <ζ> <K> <T> <in_dir> <out_dir>`
wraps the same call; set `DIFF_CKPT` to use another prior (`run_sampler_Tg.sh` also takes `GPU`).
`RADARINV_CKPT=1` enables activation checkpointing in the VQ-VAE to reduce GPU memory.

**Post-processing (points).** `experiments/fair/ours_pp.py` turns the decoded maps into points: average the
three decoder channels, keep the 8,000 cells with the highest occupancy, then remove points with fewer than two
neighbors within 0.5 m. These settings were chosen on 30 validation frames.

**K-Radar.** The same prior is applied without retraining (`experiments/kr/run_kr_final.sh`: azimuth span ±53°,
`RADARINV_AZSPAN=53`, `configs/tasks/kradar_ours_eval.yaml`, T = 500, K = 5, ζ = 0.0005; no outlier removal,
as selected on the K-Radar validation split).

**Optional switches** (environment variables, all off by default): `RADARINV_LIK=laplace` (Laplace likelihood,
Table VII), `RADARINV_CHANS=ntx,nrx` (virtual-channel subset), `RADARINV_DEBUG_DIR` (where debug images go when
`RADARINV_NODEBUG` is not 1).

## 6. Evaluation protocol

`experiments/eval_points.py <points_dir> <frame_list.txt> [label] [out.json]` scores a folder of point clouds
(`<id>.bin`, float32, x/y first) against the ground-removed LiDAR (`lidars_no_ground`), in BEV coordinates:

- **CD**: mean predicted→GT plus mean GT→predicted nearest-neighbor distance (Chamfer distance).
- **UCD**: the unidirectional predicted→GT term.
- **MHD / UMHD**: the Hausdorff distance and the directed predicted→GT Hausdorff distance (`HD` / `UMHD` in the json).
- **FID<sub>BEV</sub>**: Fréchet distance between the (x, y) coordinates of the two point sets.

## 7. Reproducing the paper

Every script in `experiments/` reads its paths from `experiments/env.sh`. Set at least `RADIAL_ROOT`,
`KRADAR_ROOT` and `PY` (the python of the `radarinv` env); outputs are written inside `experiments/`.

| Paper item | Scripts (in `experiments/`) |
|---|---|
| Table III (RADIal, 424 test frames) | Ours: `run_full.sh` (sampler on the test frames, two halves on two GPUs) → `fair/ours_pp.py` → `fair/final_table.py`. L1/L2 Reg: `fair/fair_l12.py` (uses `reg_calib.py`, λ chosen on validation). Diffradar: `run_diffradar.sh` (checkpoint chosen on 20 val frames). Radar-Diffusion: `make_radardiff_dataset.py`, `radardiff_train.sh`, `radardiff_eval.sh`, `radardiff_to_points.py`. CFAR and RadarHD use the outputs stored with RADIal (`radars_pcd`, `radars_unet`). |
| Table IV (K-Radar, 46 test frames) | `kr/kr_prep_ours.py` → `kr/run_kr_final.sh`; baselines `kr/kr_cfar.py`, `kr/kr_l12.py`; Radar-Diffusion zero-shot `kradar_radardiff.py`; table `kr/kr_table.py` (frames in `kr/subset_ids.txt`). |
| Table V (cross-scenario) | All prior: per-frame CD of Table III. Countryside prior: `sampler/run_sampler_Tg.sh` with `DIFF_CKPT=models/ldm/ldm_countryside_sd.ckpt` → `fair/eval_decoded.py`. Scenario of each frame: `research/radial_scenarios.json`. Prior training: `training/` (section 8). |
| Table VII (noise model) | Gaussian: the Table III maps. Laplace: `fair/run_laplace.sh` (`RADARINV_LIK=laplace`), scored with `fair/eval_decoded.py`. |
| Table VIII (T and K) | `sampler/run_sampler_T.sh <ζ> <K> <T> ...` on validation frames → `fair/eval_decoded.py`. |
| Fig. 3 (forward operator) | `fig3/make_fig3_bev.py` |
| Fig. 4 (RADIal qualitative) | `fair/make_fig4_cart.py` |
| Fig. 5 (K-Radar qualitative) | `make_fig5_polar.py`; Radar-Diffusion panels from `prep_kradar_fig.py` + `radardiff_kradar_fig.sh` |
| Fig. 6 (cross-scenario qualitative) | `fair/make_fig6_bev.py` (Countryside prior and the City-held-out All prior) |

Not included: the CenterPoint detection pipeline of Table VI, and the sources of Figs. 7 and 8 (latency,
convergence); the original VQ-VAE training config was not preserved (its architecture is the
`first_stage_config` in `configs/latent-diffusion/cin-ldm-vqvae-f8-radial_uncondition.yaml`).

## 8. Training a prior

`training/` holds the training code used for the additional priors (2 GPUs, effective batch 16,
22,077 optimizer steps, VQ-VAE frozen):

```bash
cd training
export RADIAL_ROOT=/path/to/RADIal/dataset_with_labels
export PYTHONPATH=$PWD:$PWD/../src/taming-transformers:$PYTHONPATH
python main.py --base configs/ldm_countryside.yaml -t --gpus 0,1 -n countryside_prior --max_steps 22077
```

`configs/ldm_countryside.yaml` expects the VQ-VAE at `../models/first_stage_models/epoch=000323.ckpt` and the
frame lists in `splits/` (Countryside training frames, 64 validation frames). Replace `data_root` with a list of
all training frames to train an all-scenario prior.

## 9. Baselines

- **Radar-Diffusion** (Zhang et al., RA-L 2024): clone [ZJU-FAST-Lab/Radar-Diffusion](https://github.com/ZJU-FAST-Lab/Radar-Diffusion),
  copy `baselines/radar_diffusion/{scripts,config}` into its `diffusion_consistency_radar/`, create the env from
  `baselines/radar_diffusion/environment_radardiff.yml`, and set `RADARDIFF_REPO` and `PY_RADARDIFF`.
- **Diffradar**: set `NANODIFF_REPO` to the Diffradar (nanoDiffusion) code with its RADIal checkpoints.
- **L1 / L2 regularized inversion**: `experiments/reg_calib.py` (calibrated operator, λ chosen per setting).

## Citation

If you use this code, please cite the paper (citation to be added on publication).
