from pathlib import Path

import datasets as ds
import numba
import numpy as np
import torch
import torch.nn as nn

_CITATION = """
@article{liao2022kitti,
    title        = {KITTI-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d},
    author       = {Liao, Yiyi and Xie, Jun and Geiger, Andreas},
    journal      = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
    volume       = 45,
    number       = 3,
    pages        = {3292--3310},
    year         = 2022,
}
"""

_SEQUENCE_SPLITS = {
    "lidargen": {
        ds.Split.TRAIN: [1, 3, 4, 5],
        ds.Split.TEST: [0, 2],
    }
}


@numba.jit(nopython=True, parallel=False)
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array


def load_points_as_images(
    point_path: str,
    scan_unfolding: bool = True,
    H: int = 64,
    W: int = 2048,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
    min_z: float = -5,
    max_z: float = 10,
):
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    xyz = points[:, :3]
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    mask_z = (z >= min_z) & (z <= max_z)

    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask_depth = (depth >= min_depth) & (depth <= max_depth)
    mask = mask_z & mask_depth
    points = np.concatenate([points[:, :4], depth, mask], axis=1)
    if scan_unfolding:
        # Assign scan rows from quadrant transitions in the ordered point cloud.
        quads = np.zeros_like(x, dtype=np.int32)
        quads[(x >= 0) & (y >= 0)] = 0  # 1st
        quads[(x < 0) & (y >= 0)] = 1  # 2nd
        quads[(x < 0) & (y < 0)] = 2  # 3rd
        quads[(x >= 0) & (y < 0)] = 3  # 4th

        diff = np.roll(quads, shift=1, axis=0) - quads
        delim_inds, _ = np.where(diff == 3)
        inds = list(delim_inds) + [len(points)]
        grid_h = np.zeros_like(x, dtype=np.int32)
        cur_ring_idx = H - 1
        for i in reversed(range(len(delim_inds))):
            grid_h[inds[i] : inds[i + 1]] = cur_ring_idx
            if cur_ring_idx >= 0:
                cur_ring_idx -= 1
            else:
                break
    else:
        # Use the vertical field of view adopted by the released experiments.
        h_up, h_down = np.deg2rad(16.5), np.deg2rad(-11.5)
        elevation = np.arcsin(z / depth) + abs(h_down)
        grid_h = 1 - elevation / (h_up - h_down)
        grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    azimuth = -np.arctan2(y, x)
    grid_w = (azimuth / np.pi + 1) / 2 % 1
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)
    grid = np.concatenate((grid_h, grid_w), axis=1)

    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W, 4 + 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])

    return proj_points.astype(np.float32)

def _points_to_bev_height_image(points):
    side_range = (-32, 32)
    forward_range = (-64, 64)
    resolution = 0.5
    height_range = (-2, 0.5)

    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    forward_mask = (x_points > forward_range[0]) & (
        x_points < forward_range[1]
    )
    side_mask = (y_points > -side_range[1]) & (y_points < -side_range[0])
    indices = np.argwhere(forward_mask & side_mask).flatten()

    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    x_image = (-y_points / resolution).astype(np.int32)
    y_image = (-x_points / resolution).astype(np.int32)
    x_image -= int(np.floor(side_range[0] / resolution))
    y_image += int(np.ceil(forward_range[1] / resolution))

    pixel_values = np.clip(z_points, height_range[0], height_range[1])
    pixel_values = scale_to_255(
        pixel_values, min=height_range[0], max=height_range[1]
    )

    width = 1 + int((side_range[1] - side_range[0]) / resolution)
    height = 1 + int((forward_range[1] - forward_range[0]) / resolution)
    image = np.zeros([height, width], dtype=np.uint8)
    image[y_image, x_image] = pixel_values
    return image[:-1, :-1]


def DealWithRadar(points):
    return _points_to_bev_height_image(points)


def DealWithLidar(points):
    return _points_to_bev_height_image(points)


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def load_points_as_BEVimages(
    point_path: str,
    scan_unfolding: bool = True,
    H: int = 128,
    W: int = 256,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
    min_z: float = -5,
    max_z: float = 10,
):
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    xyz = points[:, :3]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask_depth = (depth >= min_depth) & (depth <= max_depth)
    mask = mask_depth

    points = np.concatenate([points[:, :4], depth, mask], axis=1)

    bev_img = DealWithLidar(points)
    transposed_img = bev_img.transpose()
    bev_tensor = torch.Tensor(transposed_img)
    new_tensor_1 = bev_tensor.unsqueeze(2).repeat(1, 1, 5)
    mask_tensor = torch.ones(128, 256)
    new_tensor = torch.cat((new_tensor_1, mask_tensor.unsqueeze(2)), dim=2)

    proj_points = new_tensor.numpy()
    return proj_points.astype(np.float32)


def load_radar_points_as_BEVimages(
    point_path: str,
    scan_unfolding: bool = True,
    H: int = 128,
    W: int = 256,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
    min_z: float = -5,
    max_z: float = 10,
):
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 6))
    xyz = points[:, :3]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask_depth = (depth >= min_depth) & (depth <= max_depth)
    mask = mask_depth
    points = np.concatenate([points[:, :3], points[:, [5]], depth, mask], axis=1)

    bev_img = DealWithRadar(points)
    transposed_img = bev_img.transpose()
    bev_tensor = torch.Tensor(transposed_img)
    new_tensor_1 = bev_tensor.unsqueeze(2).repeat(1, 1, 5)
    mask_tensor = torch.ones(128, 256)
    new_tensor = torch.cat((new_tensor_1, mask_tensor.unsqueeze(2)), dim=2)

    proj_points = new_tensor.numpy()
    return proj_points.astype(np.float32)


def load_lidar_points_as_BEVfeature_images(
    file_path, voxel_generator, model_pillar, ppscatter
):
    path = str(file_path)
    if "Lidar" in path:
        points_file = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
    elif "radar" in path:
        points_file = np.fromfile(file_path, dtype=np.float32).reshape((-1, 6))
    else:
        raise ValueError(f"Cannot infer point format from path: {file_path}")

    points = torch.from_numpy(points_file[:, :3])
    voxel_output = voxel_generator.generate_voxel_with_id(points)
    voxels, coordinates, num_points, _ = voxel_output
    features = model_pillar(voxels, coordinates, num_points)
    batch_spatial_features = ppscatter(features, coordinates)
    return batch_spatial_features


class JIHE_DATA(ds.GeneratorBasedBuilder):
    """JIHE paired LiDAR-radar dataset."""

    BUILDER_CONFIGS = [
        # 64x2048
        ds.BuilderConfig(
            name="unfolding-2048",
            description="scan unfolding, 64x2048 resolution",
            data_dir="data/jihe_data/dataset",
        ),
        ds.BuilderConfig(
            name="spherical-2048",
            description="spherical projection, 64x2048 resolution",
            data_dir="data/jihe_data/dataset",
        ),
        # 64x1024
        ds.BuilderConfig(
            name="unfolding-1024",
            description="scan unfolding, 64x1024 resolution",
            data_dir="data/jihe_data/dataset",
        ),
        ds.BuilderConfig(
            name="spherical-1024",
            description="spherical projection, 64x1024 resolution",
            data_dir="data/jihe_data/dataset",
        ),
    ]

    DEFAULT_CONFIG_NAME = "spherical-1024"

    def _parse_config_name(self):
        projection, width = self.config.name.split("-")
        return projection, int(width)

    def _info(self):
        features = {
            "sample_id": ds.Value("int32"),
            "xyz": ds.Array3D((3, 128, 256), "float32"),
            "reflectance": ds.Array3D((1, 128, 256), "float32"),
            "depth": ds.Array3D((1, 128, 256), "float32"),
            "mask": ds.Array3D((1, 128, 256), "float32"),
        }
        return ds.DatasetInfo(features=ds.Features(features))

    def _split_generators(self, _):
        splits = []
        for split, subsets in _SEQUENCE_SPLITS["lidargen"].items():
            file_paths = list()
            file_cond_paths = list()
            for subset in subsets:
                wildcard = f"*_{subset:04d}_sync/Lidar/*.bin"
                wildcard_cond = f"*_{subset:04d}_sync/radar/*.bin"
                file_paths += sorted(Path(self.config.data_dir).glob(wildcard))
                file_cond_paths += sorted(
                    Path(self.config.data_dir).glob(wildcard_cond)
                )
            splits.append(
                ds.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "items": list(
                            zip(range(len(file_paths)), file_paths, file_cond_paths)
                        )
                    },
                )
            )
        return splits

    def _generate_examples(self, items):
        from spconv.pytorch.utils import PointToVoxel as VoxelGenerator

        voxel_generator = VoxelGenerator(
            vsize_xyz=[0.16, 0.16, 8],
            coors_range_xyz=[-32, -32, -1, 32, 32, 7],
            num_point_features=3,
            max_num_points_per_voxel=16,
            max_num_voxels=16000,
        )
        from tools_voxel.vfe_template import VFETemplate
        from tools_voxel.pillar_vfe import PillarVFE

        model_template = VFETemplate()
        model_pillar = PillarVFE(model_template)
        from tools_voxel.pointpillar_scatter import PointPillarScatter

        grid_size = [400, 400, 1]
        ppscatter = PointPillarScatter(grid_size)

        for sample_id, file_path, file_cond_path in items:
            bev_feature_lidar = load_lidar_points_as_BEVfeature_images(
                file_path, voxel_generator, model_pillar, ppscatter
            )
            bev_feature_radar = load_lidar_points_as_BEVfeature_images(
                file_cond_path, voxel_generator, model_pillar, ppscatter
            )

            bev_feature_lidar = bev_feature_lidar.cuda(non_blocking=True)
            bev_feature_radar = bev_feature_radar.cuda(non_blocking=True)

            yield sample_id, {
                "sample_id": sample_id,
                "xyz": bev_feature_lidar[0],
                "reflectance": bev_feature_radar[0],
                "depth": bev_feature_lidar[0],
                "mask": bev_feature_lidar[0],
            }


class PointPillarsBaseOwn(nn.Module):
    def __init__(
        self,
        num_point_features: int,
        point_cloud_range: tuple[float],
        voxel_size: tuple[float],
        use_norm: bool = True,
        with_distance: bool = False,
        use_absolute_xyz: bool = True,
        num_filters: tuple[int] | int = 64,
        num_bev_features: int = 64,
        max_num_points_per_voxel: int = 16,
        max_num_voxels: int = 16000,
    ):
        super().__init__()
        point_cloud_range = list(point_cloud_range)
        voxel_size = list(voxel_size)
        num_filters = list(num_filters)

        point_cloud_range = np.array(point_cloud_range)
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        from tools_voxel.vfe_template import VFETemplate
        from tools_voxel.pillar_vfe import PillarVFE

        self.model_template = VFETemplate()
        self.model_pillar = PillarVFE(self.model_template)
        from spconv.pytorch.utils import PointToVoxel as VoxelGenerator

        self.voxel_generator = VoxelGenerator(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features,
            max_num_points_per_voxel=max_num_points_per_voxel,
            max_num_voxels=max_num_voxels,
        )
        from tools_voxel.pointpillar_scatter import PointPillarScatter

        self.ppscatter = PointPillarScatter(grid_size)

    def forward(self, file_path, **kwargs):
        """Encode one LiDAR or radar point cloud as a BEV feature map."""
        points = self.load_lidar_points_as_BEVfeature_images(file_path)
        voxel_output = self.voxel_generator.generate_voxel_with_id(points)
        voxels, coordinates, num_points, _ = voxel_output
        features = self.model_pillar(voxels, coordinates, num_points)
        return self.ppscatter(features, coordinates)

    def load_lidar_points_as_BEVfeature_images(self, file_path):
        path = str(file_path)
        if "Lidar" in path:
            points_file = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
        elif "radar" in path:
            points_file = np.fromfile(file_path, dtype=np.float32).reshape((-1, 6))
        else:
            raise ValueError(f"Cannot infer point format from path: {file_path}")
        return torch.from_numpy(points_file[:, :3])
