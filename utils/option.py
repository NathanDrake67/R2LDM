"""Default configuration for R2LDM.

Command-line arguments in ``train_stage0.py`` and ``test.py`` override these
values. Paths are intentionally relative so that a cloned repository does not
contain machine-specific locations.
"""

import argparse
from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class ModelConfig:
    architecture: str = "efficient_unet"
    base_channels: int = 64
    temb_channels: int | None = None
    channel_multiplier: Tuple[int, int, int, int] = (1, 2, 4, 8)
    num_residual_blocks: Tuple[int, int, int, int] = (3, 3, 3, 3)
    gn_num_groups: int = 8
    gn_eps: float = 1e-6
    attn_num_heads: int = 8
    coords_encoding: Literal[
        "spherical_harmonics", "polar_coordinates", "fourier_features", None
    ] = None
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    num_training_steps: int | None = None
    num_sampling_steps: int = 128
    prediction_type: Literal["eps", "v", "x_0"] = "eps"
    loss_type: str = "l2"
    noise_schedule: str = "cosine"
    timestep_type: Literal["continuous", "discrete"] = "continuous"


@dataclass
class TrainingConfig:
    batch_size_train: int = 1
    batch_size_eval: int = 1
    num_workers: int = 1
    gradient_accumulation_steps: int = 1

    # None selects the paper training preset for the chosen stage:
    # LPCR: lr=1e-3 for 30 epochs; LVDM: lr=1e-4 for 45 epochs.
    lr: float | None = None
    max_epochs: int | None = None

    lr_warmup_steps: int = 2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    ema_decay: float = 0.995
    ema_update_every: int = 10
    mixed_precision: str = "fp16"
    dynamo_backend: str = "inductor"
    output_dir: str = "runs"
    seed: int = 3407
    save_every_epochs: int = 5

    # Stage 2 starts from the final LPCR checkpoint. A resume checkpoint, when
    # supplied, takes precedence for either stage.
    lpcr_checkpoint: str = "checkpoints/lpcr/diffusion_00030.pth"
    resume_checkpoint: str | None = None


@dataclass
class DataConfig:
    dataset: Literal["vod"] = "vod"
    root: str = "datasets/VoD"
    train_split: str = "det3d/datasets/vod/train_files_all.txt"
    val_split: str = "det3d/datasets/vod/test_files_all.txt"
    test_split: str = "det3d/datasets/vod/test_files_all.txt"
    prediction_dir: str = "outputs/predictions"

    lidar_input: str = "lidar_seq_preprocess"
    lidar_gt: str = "lidar_seq_preprocess"
    depth_format: Literal["log_depth", "inverse_depth", "depth"] = "log_depth"
    projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    train_depth: bool = True
    train_reflectance: bool = False
    resolution: Tuple[int, int] = (64, 64)
    min_depth: float = 1.45
    max_depth: float = 80.0


@dataclass
class VoxelFeatureConfig:
    """Geometry and architecture settings shared by the LPCR branches.

    The LiDAR and radar encoders must use identical settings. Change the
    values here, or pass the corresponding command-line overrides, instead of
    editing the training and inference scripts independently.
    """

    point_cloud_range: Tuple[float, float, float, float, float, float] = (
        0.0,
        -16.0,
        -2.0,
        32.0,
        16.0,
        4.0,
    )
    voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.15)
    max_points_in_voxel: int = 8
    max_voxel_num: int = 40000
    reader_num_input_features: int = 3
    backbone_num_input_features: int = 3
    backbone_ds_factor: int = 3
    neck_layer_nums: Tuple[int, int] = (5, 5)
    neck_downsample_strides: Tuple[int, int] = (1, 2)
    neck_downsample_filters: Tuple[int, int] = (128, 256)
    neck_upsample_strides: Tuple[int, int] = (1, 2)
    neck_upsample_filters: Tuple[int, int] = (256, 256)
    neck_num_input_features: int = 256
    # ``legacy_sigmoid`` reproduces the published implementation and existing
    # checkpoints. ``centered_tanh`` is the corrected signed-offset variant.
    offset_activation: Literal["legacy_sigmoid", "centered_tanh"] = (
        "legacy_sigmoid"
    )

    def grid_bounds(self) -> list[list[float]]:
        """Return the XYZ bounds expected by the diffusion model."""
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        return [[x_min, x_max], [y_min, y_max], [z_min, z_max]]


@dataclass(frozen=True)
class MetricVoxelConfig:
    """Default preprocessing grid for the reported point-cloud metrics.

    This is deliberately separate from :class:`VoxelFeatureConfig`: the
    supplied Jihe metric uses a different evaluation grid than LPCR training.
    """

    point_cloud_range: Tuple[float, float, float, float, float, float] = (
        -16.0,
        -16.0,
        -0.5,
        16.0,
        16.0,
        4.5,
    )
    voxel_size: Tuple[float, float, float] = (0.125, 0.125, 0.125)


def add_voxel_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the small set of LPCR geometry overrides shared by two entry points."""
    parser.add_argument(
        "--point-cloud-range",
        nargs=6,
        type=float,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="LPCR XYZ bounds; changing them requires training a matching model.",
    )
    parser.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        help="LPCR voxel size; changing it requires training a matching model.",
    )
    parser.add_argument(
        "--max-points-in-voxel",
        type=int,
        help="Maximum input points retained in each LPCR voxel.",
    )
    parser.add_argument(
        "--max-voxel-num",
        type=int,
        help="Maximum occupied LPCR voxels per point cloud.",
    )


def apply_voxel_config_overrides(
    cfg: "Config", args: argparse.Namespace
) -> None:
    """Apply optional CLI voxel overrides to a project configuration."""
    voxel_cfg = cfg.voxelextractor
    if args.point_cloud_range is not None:
        voxel_cfg.point_cloud_range = tuple(args.point_cloud_range)
    if args.voxel_size is not None:
        voxel_cfg.voxel_size = tuple(args.voxel_size)
    if args.max_points_in_voxel is not None:
        voxel_cfg.max_points_in_voxel = args.max_points_in_voxel
    if args.max_voxel_num is not None:
        voxel_cfg.max_voxel_num = args.max_voxel_num


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    voxelextractor: VoxelFeatureConfig = field(default_factory=VoxelFeatureConfig)


STAGE_PRESETS = {
    "lpcr": {"stage_id": 0, "lr": 1e-3, "epochs": 30},
    "ldm": {"stage_id": 1, "lr": 1e-4, "epochs": 45},
}
