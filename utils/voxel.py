"""Shared LPCR voxelization and sparse-encoder construction helpers.

The sparse dependencies are imported inside the factory functions so that the
command-line help for training and inference remains available without spconv.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from utils.option import VoxelFeatureConfig


def validate_voxel_config(config: VoxelFeatureConfig) -> None:
    """Reject geometry settings that would create an ambiguous voxel grid."""
    if len(config.point_cloud_range) != 6:
        raise ValueError("point_cloud_range must contain six values")
    if len(config.voxel_size) != 3:
        raise ValueError("voxel_size must contain three values")
    if config.max_points_in_voxel <= 0:
        raise ValueError("max_points_in_voxel must be positive")
    if config.max_voxel_num <= 0:
        raise ValueError("max_voxel_num must be positive")

    lower = config.point_cloud_range[:3]
    upper = config.point_cloud_range[3:]
    for axis, (minimum, maximum, size) in enumerate(
        zip(lower, upper, config.voxel_size, strict=True)
    ):
        if size <= 0:
            raise ValueError(f"voxel_size[{axis}] must be positive")
        if maximum <= minimum:
            raise ValueError(
                f"point_cloud_range upper bound must exceed lower bound on axis {axis}"
            )
        cells = (maximum - minimum) / size
        if not math.isclose(cells, round(cells), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "Each point-cloud extent must be divisible by its voxel size; "
                f"axis {axis} gives {cells} cells."
            )


def build_voxelizer(config: VoxelFeatureConfig):
    """Build the stateless point-cloud-to-voxel adapter used by LPCR."""
    validate_voxel_config(config)

    from s2dense.voxelization import Voxelization

    return Voxelization(
        range=list(config.point_cloud_range),
        voxel_size=list(config.voxel_size),
        max_points_in_voxel=config.max_points_in_voxel,
        max_voxel_num=config.max_voxel_num,
    )


def validate_checkpoint_voxel_geometry(
    saved_config: Mapping[str, object], config: VoxelFeatureConfig
) -> None:
    """Ensure a checkpoint agrees with any stored LPCR geometry fields.

    Older checkpoints did not store these fields, so omitted values remain
    supported. New checkpoints fail early instead of silently mixing a model
    with an incompatible voxel grid.
    """
    fields = (
        "point_cloud_range",
        "voxel_size",
        "max_points_in_voxel",
        "max_voxel_num",
    )
    for field in fields:
        if field not in saved_config:
            continue
        saved_value = saved_config[field]
        current_value = getattr(config, field)
        normalized_saved_value = (
            tuple(saved_value)
            if isinstance(current_value, tuple)
            and isinstance(saved_value, (list, tuple))
            else saved_value
        )
        if normalized_saved_value != current_value:
            raise ValueError(
                f"Checkpoint {field} is {normalized_saved_value!r}, but the current "
                f"configuration uses {current_value!r}. Use matching voxel "
                "settings or retrain the model."
            )


def build_lpcr_encoder(config: VoxelFeatureConfig):
    """Build one independent sparse LPCR encoder branch."""
    validate_voxel_config(config)

    from det3d.models.backbones.scn import SpMiddleResNetFHD
    from det3d.models.necks.rpn import S2D_RPN
    from s2dense.voxel_encoder import VoxelFeatureExtractorV3
    from s2dense.voxelnet import KD_VoxelNet

    reader = VoxelFeatureExtractorV3(
        num_input_features=config.reader_num_input_features
    )
    backbone = SpMiddleResNetFHD(
        num_input_features=config.backbone_num_input_features,
        ds_factor=config.backbone_ds_factor,
    )
    neck = S2D_RPN(
        layer_nums=list(config.neck_layer_nums),
        ds_layer_strides=list(config.neck_downsample_strides),
        ds_num_filters=list(config.neck_downsample_filters),
        us_layer_strides=list(config.neck_upsample_strides),
        us_num_filters=list(config.neck_upsample_filters),
        num_input_features=config.neck_num_input_features,
    )
    return KD_VoxelNet(
        reader=reader,
        backbone=backbone,
        neck=neck,
        with_neck=False,
        offset_activation=config.offset_activation,
    )


def build_lpcr_encoder_pair(config: VoxelFeatureConfig, device):
    """Build independent LiDAR and radar LPCR branches on ``device``."""
    lidar_encoder = build_lpcr_encoder(config).to(device)
    radar_encoder = build_lpcr_encoder(config).to(device)
    return lidar_encoder, radar_encoder
