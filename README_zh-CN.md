# R2LDM

**R2LDM: An Efficient 4D Radar Super-Resolution Framework Leveraging Diffusion Model**（IROS 2025）官方实现。

[[GitHub](https://github.com/NathanDrake67/R2LDM)] [[arXiv](https://arxiv.org/abs/2503.17097)] [[DOI](https://doi.org/10.1109/IROS60139.2025.11246739)] [[English README](README.md)] [[许可证](LICENSE)]

R2LDM 从稀疏的 4D 毫米波雷达点云生成稠密、类似 LiDAR 的点云。方法以体素特征统一表示雷达与 LiDAR 点云，使用潜在体素扩散模型（LVDM）完成条件生成，并通过潜在点云重建模块（LPCR）恢复三维点。

> **发布状态：** 源代码以 Apache-2.0 许可证发布。当前尚未包含数据预处理工具和托管的预训练权重；精确复现论文仍需完成 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) 中的事项。

## 方法与训练流程

训练分为两个阶段：

| 阶段 | 训练内容 | 学习率 | Epoch | 初始化方式 |
| --- | --- | ---: | ---: | --- |
| 1. LPCR | LiDAR 体素编码器与点云重建模块 | `1e-3` | 30 | 从头训练 |
| 2. LVDM | 雷达编码器与条件潜在扩散模型 | `1e-4` | 45 | 加载第一阶段权重，并冻结 LiDAR/LPCR 分支 |

论文报告了 6-10 倍的雷达点云稠密化效果，点云配准召回率最高提升 31.7%，目标检测精度最高提升 24.9%。

## 环境安装

本实现需要 Python 3.10、PyTorch 2.1、CUDA 和 NVIDIA GPU。论文实验使用单张 RTX 2080 Ti，batch size 为 1。

```bash
conda env create -f environment.yaml
conda activate r2ldm
```

环境文件会安装 `spconv-cu120==2.3.6`。如果预编译包不支持你的平台或显卡，请参考 [spconv 官方安装说明](https://github.com/traveller59/spconv#install)从源码构建。

## 数据准备

当前代码需要预处理后的 View-of-Delft 风格数据：

- LiDAR `.bin`：小端 `float32`，形状为 `N x 4`；
- Radar `.bin`：`float32`，形状为 `N x 7`；
- 当前模型读取两者的 XYZ 三列。

目录结构如下：

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

划分文件每行格式为 `<sequence>/<frame>`，例如 `03/00042`。默认划分文件位于 `det3d/datasets/vod/`，也可以通过命令行覆盖。

论文使用 Patchwork++ 去除 LiDAR 地面点，并在体素化前对齐雷达与 LiDAR 的视场。现有代码快照不包含这些预处理工具；如需精确复现论文结果，公开发布时还需补充它们。

## 训练

默认配置位于 [`utils/option.py`](utils/option.py)，命令行参数可覆盖路径和训练计划。

### 第一阶段：LPCR

```bash
python train_stage0.py \
  --stage lpcr \
  --data-root /path/to/VODData
```

默认学习率为 `1e-3`，训练 30 个 epoch。权重保存在：

```text
runs/vod/spherical-1024/lpcr/<timestamp>/models/
```

### 第二阶段：潜在扩散模型

```bash
python train_stage0.py \
  --stage ldm \
  --data-root /path/to/VODData \
  --lpcr-checkpoint /path/to/diffusion_00030.pth
```

默认学习率为 `1e-4`，训练 45 个 epoch。若需续训，使用同一阶段并添加 `--resume /path/to/checkpoint.pth`。可运行 `python train_stage0.py --help` 查看全部参数。

### LPCR 偏置参数化

论文实现使用 `sigmoid` 预测点偏置；为复现已有 checkpoint，默认保留为 `legacy_sigmoid`。但代码中的监督目标是 `point - voxel_center`，它是有正有负的。因此新实验建议添加 `--offset-activation centered_tanh`，使用 `tanh(raw) * [dx/2, dy/2, dz/2]`，并且只在非空体素上监督偏置。训练和测试必须使用相同模式；改用该模式后，需要重新训练 LPCR 和 LVDM。

## 推理

```bash
python test.py \
  --checkpoint /path/to/stage2_checkpoint.pth \
  --data-root /path/to/VODData \
  --output-dir outputs/predictions
```

默认使用论文中综合表现较好的 128 个 DDIM 采样步。输出为 `float32` XYZ `.bin` 文件，并保留数据集划分与序列目录结构。快速检查可添加 `--max-samples 2`。

若 checkpoint 使用 `centered_tanh` 训练，测试时也必须添加 `--offset-activation centered_tanh`。

## 精度指标

你提供的指标脚本已整理为可配置入口，可计算平方 L2 Chamfer Distance、平方 L2 Hausdorff Distance、F-score，以及 BEV JSD/MMD：

```bash
python eval_metric_2025.py \
  --data-root /path/to/evaluation_root \
  --sequences zhengda4qi_1 zhengda4qi_3
```

目录格式为 `evaluation_root/gen/<sequence>/*.bin` 与 `evaluation_root/gt/<sequence>/*.bin` 一一对应。当前 `test.py` 输出 XYZ 三列，因此默认按三列读取；旧版四列生成文件请添加 `--gen-cols 4`。如需 PointNet FPD，可添加 `--pointnet-fpd`，首次运行会下载预训练特征提取器。

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

## 致谢与许可证

本实现包含受 [Sparse2Dense](https://github.com/stevewongv/Sparse2Dense) 与 [CenterPoint](https://github.com/tianweiy/CenterPoint) 启发并修改的组件，并依赖 [spconv](https://github.com/traveller59/spconv)。重新发布前请阅读 [THIRD_PARTY.md](THIRD_PARTY.md)。

R2LDM 采用 [Apache License 2.0](LICENSE) 发布。第三方组件继续受其各自许可证约束，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
