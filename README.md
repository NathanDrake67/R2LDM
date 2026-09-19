# R2LDM

Official implementation of **R2LDM: An Efficient 4D Radar Super-Resolution Framework Leveraging Diffusion Model** (IROS 2025).

<p align="center">
  <img
    src="https://raw.githubusercontent.com/NathanDrake67/zhengby.github.io/master/images/R2LDM.gif"
    width="600"
    alt="R2LDM demo"
  />
</p>

[[GitHub](https://github.com/NathanDrake67/R2LDM)] [[arXiv](https://arxiv.org/abs/2503.17097)] [[DOI](https://doi.org/10.1109/IROS60139.2025.11246739)] [[Chinese README](README_zh-CN.md)] [[License](LICENSE)]

R2LDM generates dense, LiDAR-like point clouds from sparse 4D radar point clouds. It uses voxel features as the shared representation, a Latent Voxel Diffusion Model (LVDM) for conditional generation, and a Latent Point Cloud Reconstruction (LPCR) module to recover 3D points.

> **Release status:** the source code is released under Apache-2.0. Dataset preprocessing utilities and hosted checkpoints are not yet included, so exact paper reproduction still requires the items tracked in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## Method
<p align="center">
  <img
    src="https://raw.githubusercontent.com/NathanDrake67/zhengby.github.io/master/images/teaser.png" 
    width="600"
    alt="R2LDM demo"
  />
</p>



## Installation

The implementation requires Python 3.10, PyTorch 2.1, CUDA, and an NVIDIA GPU. The paper experiments used a single RTX 2080 Ti with batch size 1.

```bash
conda env create -f environment.yaml
conda activate r2ldm
```

`spconv-cu120==2.3.6` is installed from PyPI. CUDA 12.0 and 12.1 are compatible at the minor-version level when the installed NVIDIA driver is sufficiently recent. If the prebuilt wheel does not support your platform or GPU, follow the upstream [spconv build instructions](https://github.com/traveller59/spconv#install).

## Data preparation

This code snapshot expects preprocessed View-of-Delft-style data. LiDAR files are little-endian `float32` arrays with shape `N x 4`; radar files are `float32` arrays with shape `N x 7`. R2LDM currently consumes the XYZ columns.

Expected directory structure:

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

Each split file contains one sample per line in the form `<sequence>/<frame>`, for example `03/00042`. Default split files are under `det3d/datasets/vod/` and can be overridden on the command line.

The paper removes LiDAR ground points with Patchwork++ and aligns the radar/LiDAR field of view before voxelization. Those preprocessing utilities are not present in the supplied code snapshot and must be released separately for exact reproduction.

## Training

All defaults live in [`utils/option.py`](utils/option.py). Command-line options override path and schedule settings.

### Stage 1: LPCR

```bash
python train_stage0.py \
  --stage lpcr \
  --data-root /path/to/VODData
```

This uses `lr=1e-3` and 30 epochs by default. Checkpoints are written under:

```text
runs/vod/spherical-1024/lpcr/<timestamp>/models/
```

Place or copy the final checkpoint at `checkpoints/lpcr/diffusion_00030.pth`, or pass its path explicitly in Stage 2.

### Stage 2: latent diffusion model

```bash
python train_stage0.py \
  --stage ldm \
  --data-root /path/to/VODData \
  --lpcr-checkpoint /path/to/diffusion_00030.pth
```

This uses `lr=1e-4` and 45 epochs by default. Use `--lr` or `--epochs` only for ablations. To continue an interrupted run, pass `--resume /path/to/checkpoint.pth` with the same stage.

Run `python train_stage0.py --help` for all path and schedule options.

### LPCR offset parameterization

The paper implementation predicts point offsets with `sigmoid`, and this is retained as the default `legacy_sigmoid` mode so existing checkpoints remain reproducible. The actual supervision in the code is `point - voxel_center`, which is signed. For a new corrected experiment, use `--offset-activation centered_tanh`; it predicts `tanh(raw) * [dx/2, dy/2, dz/2]` and supervises offsets only in occupied voxels. Do not mix the two modes across training and inference. The corrected mode requires retraining LPCR and LVDM.

## Inference

```bash
python test.py \
  --checkpoint /path/to/stage2_checkpoint.pth \
  --data-root /path/to/VODData \
  --output-dir outputs/predictions
```

The default is 128 DDIM sampling steps, matching the balanced setting reported in the paper. Generated point clouds are written as `float32` XYZ `.bin` files while preserving the dataset split and sequence layout. For a quick smoke test, add `--max-samples 2`.

If the checkpoint was trained with the corrected LPCR mode, also pass `--offset-activation centered_tanh`.

## Metrics

The supplied evaluation code has been cleaned into a configurable entry point. It computes squared-L2 Chamfer and Hausdorff distances, F-score, and BEV JSD/MMD:

```bash
python eval_metric_2025.py \
  --data-root /path/to/evaluation_root \
  --sequences zhengda4qi_1 zhengda4qi_3
```

The expected layout is `evaluation_root/gen/<sequence>/*.bin` paired with `evaluation_root/gt/<sequence>/*.bin`. Current `test.py` outputs XYZ files, so the default is three columns. Use `--gen-cols 4` for legacy generated files containing an extra channel. Add `--pointnet-fpd` to compute the optional PointNet Frechet distance; this downloads the pretrained extractor on first use.

## Repository layout

```text
R2LDM/
├── train_stage0.py             # canonical two-stage training implementation
├── test.py                     # canonical point-cloud generation/test implementation
├── train.py                    # optional training alias
├── inference.py                # optional inference alias
├── eval_metric_2025.py         # point-cloud accuracy metrics
├── utils/option.py             # default configuration and stage presets
├── models/                     # diffusion model and Efficient U-Net
├── s2dense/                    # voxel encoder and LPCR components
├── det3d/                      # adapted sparse 3D backbone and data utilities
├── metrics/                    # point-cloud distribution metrics
└── environment.yaml
```

Large checkpoints, datasets, generated point clouds, compiled CUDA artifacts, and copied `apex`/`spconv` source trees are intentionally excluded from the repository.

## Citation

If you find this work useful, please cite:

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

## Acknowledgements

This implementation contains adapted components inspired by [Sparse2Dense](https://github.com/stevewongv/Sparse2Dense) and [CenterPoint](https://github.com/tianweiy/CenterPoint), and depends on [spconv](https://github.com/traveller59/spconv). See [THIRD_PARTY.md](THIRD_PARTY.md) before redistribution.

## License

R2LDM is released under the [Apache License 2.0](LICENSE). Third-party components remain subject to their respective licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).
