# R2LDM: An Efficient 4D Radar Super-Resolution Framework Leveraging Diffusion Model（IROS 2025）官方实现


<p align="center">
  <img
    src="https://raw.githubusercontent.com/NathanDrake67/zhengby.github.io/master/images/R2LDM.gif"
    width="600"
    alt="R2LDM 点云生成演示"
  />
</p>

<p align="center">
  <a href="https://github.com/NathanDrake67/R2LDM">GitHub</a>
  · <a href="https://arxiv.org/abs/2503.17097">arXiv</a>
  · <a href="https://doi.org/10.1109/IROS60139.2025.11246739">DOI</a>
  · <a href="README.md">English README</a>
  · <a href="LICENSE">Apache-2.0</a>
</p>

R2LDM 从稀疏的 4D 雷达点云生成稠密、类似 LiDAR 的点云。方法以体素特征统一表示雷达与 LiDAR 点云，使用潜在体素扩散模型（Latent Voxel Diffusion Model, LVDM）完成条件生成，并通过潜在点云重建模块（Latent Point Cloud Reconstruction, LPCR）恢复三维点。

<p align="center">
  <img
    src="https://raw.githubusercontent.com/NathanDrake67/zhengby.github.io/master/images/teaser.png"
    width="760"
    alt="R2LDM 方法概览"
  />
</p>

## 发布范围

本仓库包含两阶段训练、推理和可配置的点云指标代码；不包含数据集、训练权重、生成点云、编译后的 CUDA 产物以及第三方依赖的源码副本。

论文使用的完整预处理流程、可下载 checkpoint 和部分数据集实验资产尚未包含。若需要宣称精确复现论文或重新分发额外资产，请先阅读 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)。

## 环境安装

当前代码面向 Python 3.10、PyTorch 2.1、CUDA 与 NVIDIA GPU。

```bash
conda env create -f environment.yaml
conda activate r2ldm
```

环境会安装 `spconv-cu120==2.3.6`。训练和推理需要支持 CUDA 的 NVIDIA GPU，以及足够新的 NVIDIA 驱动。

## 数据准备

当前入口脚本需要预处理后的 View-of-Delft 风格数据：

- LiDAR 文件：`float32`，形状为 `N x 4`；
- Radar 文件：`float32`，形状为 `N x 7`；
- R2LDM 从两者中读取 XYZ 三列。

推荐目录结构：

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

划分文件每行使用 `<sequence>/<frame>` 格式。默认路径位于 [`utils/option.py`](utils/option.py)，也可通过命令行参数覆盖。

> **重要说明：** 论文在体素化前进行了地面去除、雷达/LiDAR 视场对齐和多帧雷达聚合。当前源码不含这些完整预处理工具。论文中的 VoD 体素设置也不同于当前代码默认值；复用已有 checkpoint 时，请勿随意修改体素网格。

## 训练

### 论文中报告的训练设置

正式论文报告：在单张 NVIDIA RTX 2080 Ti、batch size 为 1、AdamW 优化器下，训练设置如下。

| 阶段 | 训练模块 | 初始学习率 | 论文报告的训练时长 | 默认 epoch 数 |
| --- | --- | ---: | ---: | --- |
| 阶段 1 | LPCR | `1e-3` | 约 9 小时 | 30 |
| 阶段 2 | LVDM | `1e-4` | 约 21 小时 | 45 |

本仓库在 `STAGE_PRESETS` 中使用 LPCR 30 个 epoch、LVDM 45 个 epoch 作为代码默认值；
### 阶段 1：LPCR

```bash
python train_stage0.py \
  --stage lpcr \
  --data-root /path/to/VoD
```

仓库默认训练计划为：`lr=1e-3`、30 个 epoch。checkpoint 默认保存到：

```text
runs/vod/spherical-1024/lpcr/<timestamp>/models/
```

### 阶段 2：潜在体素扩散模型

```bash
python train_stage0.py \
  --stage ldm \
  --data-root /path/to/VoD \
  --lpcr-checkpoint /path/to/stage1_checkpoint.pth
```

仓库默认训练计划为：`lr=1e-4`、45 个 epoch。第二阶段从第一阶段 checkpoint 初始化，并冻结 LiDAR/LPCR 分支。若训练中断，可为同一阶段传入 `--resume /path/to/checkpoint.pth` 续训。

### 统一的 LPCR 配置

[`utils/option.py`](utils/option.py) 是 LPCR 体素网格和稀疏编码器参数的唯一配置来源。训练与推理共同调用 [`utils/voxel.py`](utils/voxel.py) 的构建函数，因此 LiDAR 与 Radar 分支保持一致。

单次实验可仅覆盖必要的几何参数：

```bash
python train_stage0.py --stage lpcr \
  --point-cloud-range 0 -16 -2 32 16 4 \
  --voxel-size 0.1 0.1 0.15 \
  --max-points-in-voxel 8 \
  --max-voxel-num 40000
```

修改点云范围或体素大小后，必须重新训练与之匹配的 LPCR 和 LVDM checkpoint。新的 checkpoint 会保存这些参数；推理时若 checkpoint 中存在这些字段，代码会自动验证一致性。

### LPCR 偏置模式

默认的 `legacy_sigmoid` 对应论文实现和已有 checkpoint 的行为。`centered_tanh` 是供重新训练模型使用的有符号偏置实验性修正。训练和推理不能混用两种模式；切换模式后必须重新训练两个阶段。

运行 `python train_stage0.py --help` 可查看全部训练参数。

## 推理

```bash
python test.py \
  --checkpoint /path/to/stage2_checkpoint.pth \
  --data-root /path/to/VoD \
  --output-dir outputs/predictions
```

默认使用 128 个 DDIM 采样步。论文将 128 步选为综合表现平衡的推理设置。生成结果为 `float32` XYZ `.bin` 文件，并保持原有数据划分与序列目录结构。可添加 `--max-samples 2` 做简短检查。

如有需要，请传入与 checkpoint 训练时相同的 LPCR 参数，例如 `--offset-activation centered_tanh` 或对应的体素网格覆盖参数。

## 精度指标

当前指标入口为 [`eval_metric_2025.py`](eval_metric_2025.py)，可对配对点云计算平方 L2 Chamfer Distance、平方 L2 Hausdorff Distance、F-score 与 BEV JSD/MMD：

```bash
python eval_metric_2025.py \
  --data-root /path/to/evaluation_root \
  --sequences <sequence_1> <sequence_2>
```

目录结构：

```text
evaluation_root/
├── gen/<sequence>/<frame>.bin
└── gt/<sequence>/<frame>.bin
```

`test.py` 输出 XYZ 三列，因此指标脚本默认按三列读取。旧版四列生成文件请添加 `--gen-cols 4`。评估体素网格由 `MetricVoxelConfig` 单独定义，故意不与 LPCR 训练网格绑定。仅在适合你的评估协议时再启用 `--pointnet-fpd`。

## 仓库结构

```text
R2LDM/
├── train_stage0.py       # 两阶段 LPCR/LVDM 训练
├── test.py               # 点云生成
├── eval_metric_2025.py   # 点云指标
├── utils/option.py       # 训练计划与统一配置
├── utils/voxel.py        # LPCR 体素化/编码器构建函数
├── models/               # 扩散模型与 Efficient U-Net
├── s2dense/              # 体素编码器与 LPCR 组件
├── det3d/                # 适配后的稀疏 3D 骨干与数据工具
├── metrics/              # 点云指标
└── environment.yaml
```

## 引用

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

## 许可证与致谢

R2LDM 使用 [Apache License 2.0](LICENSE) 发布。代码中包含改编的研究组件，并依赖 Sparse2Dense、CenterPoint/Det3D 和 spconv 等第三方软件。重新分发前请阅读 [THIRD_PARTY.md](THIRD_PARTY.md)。
