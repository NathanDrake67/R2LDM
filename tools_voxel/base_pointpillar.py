"""PointPillars-style point-cloud encoders used by legacy data pipelines."""

import numpy as np
import torch
from torch import nn


class _BasePointCloudEncoder(nn.Module):
    def __init__(
        self,
        num_point_features: int,
        point_cloud_range: tuple[float, ...],
        voxel_size: tuple[float, ...],
        use_norm: bool = True,
        with_distance: bool = False,
        use_absolute_xyz: bool = True,
        num_filters: tuple[int, ...] | int = 64,
        num_bev_features: int = 64,
        max_num_points_per_voxel: int = 16,
        max_num_voxels: int = 16000,
    ):
        super().__init__()

        point_cloud_range = np.asarray(point_cloud_range)
        voxel_size = list(voxel_size)
        grid_size = np.round(
            (point_cloud_range[3:6] - point_cloud_range[0:3])
            / np.asarray(voxel_size)
        ).astype(np.int64)

        from spconv.pytorch.utils import PointToVoxel as VoxelGenerator
        from tools_voxel.pillar_vfe import PillarVFE
        from tools_voxel.pointpillar_scatter import PointPillarScatter
        from tools_voxel.vfe_template import VFETemplate

        self.model_template = VFETemplate()
        self.model_pillar = PillarVFE(self.model_template)
        self.voxel_generator = VoxelGenerator(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features,
            max_num_points_per_voxel=max_num_points_per_voxel,
            max_num_voxels=max_num_voxels,
        )
        self.ppscatter = PointPillarScatter(grid_size)

    def forward(self, file_path, **kwargs):
        """Encode one LiDAR or radar point cloud as a BEV feature map."""
        points = self.load_lidar_points_as_BEVfeature_images(file_path)
        voxels, coordinates, num_points, _ = (
            self.voxel_generator.generate_voxel_with_id(points)
        )
        features = self.model_pillar(voxels, coordinates, num_points)
        return self.ppscatter(features, coordinates)

    @staticmethod
    def load_lidar_points_as_BEVfeature_images(file_path):
        path = str(file_path)
        if "Lidar" in path:
            num_columns = 4
        elif "radar" in path:
            num_columns = 6
        else:
            raise ValueError(f"Cannot infer point format from path: {file_path}")

        points = np.fromfile(file_path, dtype=np.float32).reshape((-1, num_columns))
        return torch.from_numpy(points[:, :3])


class PointPillarsBaseOwn(_BasePointCloudEncoder):
    """PointPillars-compatible BEV encoder."""


class VoxelNetBaseOwn(_BasePointCloudEncoder):
    """Legacy alias for the same BEV encoder implementation."""
