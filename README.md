# R2LDM

Official implementation of **R2LDM: An Efficient 4D Radar Super-Resolution Framework Leveraging Diffusion Model** (IROS 2025).

<p align="center">
  <img
    src="https://raw.githubusercontent.com/NathanDrake67/zhengby.github.io/master/images/R2LDM.gif"
    width="600"
    alt="R2LDM point-cloud generation demo"
  />
</p>

<p align="center">
  <a href="https://github.com/NathanDrake67/R2LDM">GitHub</a>
  · <a href="https://arxiv.org/abs/2503.17097">arXiv</a>
  · <a href="https://doi.org/10.1109/IROS60139.2025.11246739">DOI</a>
  · <a href="README_zh-CN.md">中文文档</a>
  · <a href="LICENSE">Apache-2.0</a>
</p>

R2LDM generates dense, LiDAR-like point clouds from sparse 4D radar point clouds. It represents radar and LiDAR using voxel features, learns conditional generation with a Latent Voxel Diffusion Model (LVDM), and reconstructs 3D points with a Latent Point Cloud Reconstruction (LPCR) module.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/NathanDrake67/zhengby.github.io/master/images/teaser.png"
    width="760"
    alt="R2LDM overview"
  />
</p>

## Release scope

This repository contains the two-stage training code, inference code, and configurable point-cloud metrics. It intentionally excludes datasets, training checkpoints, generated point clouds, compiled CUDA artifacts, and copied dependency source trees.

The exact paper preprocessing pipeline, released checkpoints, and some dataset-specific experiment assets are not included yet. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) before claiming an exact paper reproduction or redistributing additional assets.

## Installation

The released code targets Python 3.10, PyTorch 2.1, CUDA, and an NVIDIA GPU.

```bash
conda env create -f environment.yaml
conda activate r2ldm
```

The environment installs `spconv-cu120==2.3.6`. A CUDA-capable GPU and a sufficiently recent NVIDIA driver are required for training and inference.

## Data preparation

The current entry points expect preprocessed View-of-Delft-style data:

- LiDAR files: little-endian `float32`, shaped `N x 4`;
- radar files: `float32`, shaped `N x 7`;
- R2LDM consumes the XYZ columns from both files.

An expected layout is:

```text
datasets/VoD/
├── train/
│   └── <sequence>/
│       ├── radar/<frame>.bin
│       └── lidar_seq_preprocess/<frame>.bin
├── val/
│   └── <sequence>/...
├── test/
│   └── <sequence>/...
└── test_rest/
    └── <sequence>/...
```

Split files contain one `<sequence>/<frame>` entry per line. Their default paths are configured in [`utils/option.py`](utils/option.py) and can be overridden from the command line.

> **Important:** the paper applies ground removal, radar/LiDAR field-of-view alignment, and multi-frame radar aggregation before voxelization. Those exact preprocessing utilities are not part of this source release. The paper's VoD voxel setting also differs from the current code default; do not change the voxel grid while reusing an existing checkpoint.

## Training

### Paper-reported recipe

The IROS paper reports the following implementation details on one NVIDIA RTX 2080 Ti with batch size 1 and AdamW:

| Stage | Module trained | Initial learning rate | Reported training time | Epoch count in paper |
| --- | --- | ---: | ---: | --- |
| 1 | LPCR | `1e-3` | about 9 hours | Not reported |
| 2 | LVDM | `1e-4` | about 21 hours | Not reported |

The paper does **not** specify epoch counts. The public implementation uses 30 epochs for LPCR and 45 epochs for LVDM as repository defaults in `STAGE_PRESETS`; these are code defaults, not values claimed by the paper.

### Stage 1: LPCR

```bash
python train_stage0.py \
  --stage lpcr \
  --data-root /path/to/VoD
```

Default repository schedule: `lr=1e-3`, 30 epochs. Checkpoints are saved under:

```text
runs/vod/spherical-1024/lpcr/<timestamp>/models/
```

### Stage 2: latent voxel diffusion

```bash
python train_stage0.py \
  --stage ldm \
  --data-root /path/to/VoD \
  --lpcr-checkpoint /path/to/stage1_checkpoint.pth
```

Default repository schedule: `lr=1e-4`, 45 epochs. Stage 2 initializes from the Stage 1 checkpoint and freezes the LiDAR/LPCR branch. Resume an interrupted run with `--resume /path/to/checkpoint.pth` for the same stage.

### Shared LPCR configuration

[`utils/option.py`](utils/option.py) is the single source of truth for the LPCR voxel grid and sparse-encoder settings. Both training and inference use the shared factory in [`utils/voxel.py`](utils/voxel.py), so the LiDAR and radar branches stay consistent.

For a one-off experiment, override only the relevant geometry:

```bash
python train_stage0.py --stage lpcr \
  --point-cloud-range 0 -16 -2 32 16 4 \
  --voxel-size 0.1 0.1 0.15 \
  --max-points-in-voxel 8 \
  --max-voxel-num 40000
```

Changing the point-cloud range or voxel size requires training matching LPCR and LVDM checkpoints. New checkpoints store these values, and inference validates them when they are available.

### LPCR offset mode

`legacy_sigmoid` is the default and corresponds to the published implementation/checkpoint behavior. `centered_tanh` is an experimental signed-offset correction for newly trained models. Do not mix the two modes between training and inference; changing modes requires retraining both stages.

Run `python train_stage0.py --help` for all training options.

## Inference

```bash
python test.py \
  --checkpoint /path/to/stage2_checkpoint.pth \
  --data-root /path/to/VoD \
  --output-dir outputs/predictions
```

The default uses 128 DDIM sampling steps. The paper selected 128 steps as its balanced inference configuration. Generated files are `float32` XYZ `.bin` files that preserve the split and sequence layout. Add `--max-samples 2` for a short smoke test.

Pass the same LPCR options used to train the checkpoint when needed, for example `--offset-activation centered_tanh` or matching voxel-grid overrides.

## Evaluation

The current metric entry point is [`eval_metric_2025.py`](eval_metric_2025.py). It reports squared-L2 Chamfer distance, squared-L2 Hausdorff distance, F-score, and BEV JSD/MMD for paired point clouds:

```bash
python eval_metric_2025.py \
  --data-root /path/to/evaluation_root \
  --sequences <sequence_1> <sequence_2>
```

Expected layout:

```text
evaluation_root/
├── gen/<sequence>/<frame>.bin
└── gt/<sequence>/<frame>.bin
```

`test.py` writes XYZ files, so the metric defaults to three columns. Use `--gen-cols 4` for legacy four-column generated files. The evaluation voxel grid has a separate default in `MetricVoxelConfig`; it is intentionally independent of the LPCR training grid. Add `--pointnet-fpd` only when the optional PointNet feature extractor is appropriate for your evaluation.

## Repository layout

```text
R2LDM/
├── train_stage0.py       # two-stage LPCR/LVDM training
├── test.py               # point-cloud generation
├── eval_metric_2025.py   # point-cloud metrics
├── utils/option.py       # schedules and shared configuration
├── utils/voxel.py        # LPCR voxel/encoder factory
├── models/               # diffusion model and Efficient U-Net
├── s2dense/              # voxel encoder and LPCR components
├── det3d/                # adapted sparse 3D backbone and data utilities
├── metrics/              # point-cloud metrics
└── environment.yaml
```

## Citation

```bibtex
@inproceedings{zheng2025r2ldm,
  title     = {R2LDM: An Efficient 4D Radar Super-Resolution Framework Leveraging Diffusion Model},
  author    = {Zheng, Boyuan and Lu, Shouyi and Huang, Renbo and Huang, Minqing and Lu, Fan and Tian, Wei and Zhuo, Guirong and Xiong, Lu},
  booktitle = {2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {769--776},
  year      = {2025},
  doi       = {10.1109/IROS60139.2025.11246739}
}
```

## License and acknowledgements

R2LDM is released under the [Apache License 2.0](LICENSE). The code contains adapted research components and relies on third-party software, including Sparse2Dense, CenterPoint/Det3D, and spconv. Review [THIRD_PARTY.md](THIRD_PARTY.md) before redistribution.
