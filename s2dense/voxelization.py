"""Voxelization adapter used by the training and inference entry points."""

import numpy as np
import torch

from det3d.datasets.pipelines.formating import Reformat
from s2dense.voxel_generator import VoxelGenerator


_TENSOR_KEYS = {
    "voxels",
    "num_points",
    "num_gt",
    "voxel_labels",
    "num_voxels",
    "cyv_voxels",
    "cyv_num_points",
    "cyv_num_voxels",
    "dense_voxels",
    "dense_num_points",
    "dense_num_voxels",
    "reconstruction_voxels",
    "reconstruction_num_points",
    "reconstruction_num_voxels",
    "reconstruction_voxels_4",
    "reconstruction_num_points_4",
    "reconstruction_num_voxels_4",
    "reconstruction_voxels_2",
    "reconstruction_num_points_2",
    "reconstruction_num_voxels_2",
}

_COORDINATE_KEYS = {
    "coordinates",
    "points",
    "cyv_coordinates",
    "dense_points",
    "dense_coordinates",
    "reconstruction_coordinates",
    "reconstruction_points",
    "reconstruction_coordinates_4",
    "reconstruction_coordinates_2",
}

_CUDA_KEYS = {
    "voxels",
    "dense_voxels",
    "bev_map",
    "coordinates",
    "dense_coordinates",
    "num_points",
    "dense_num_points",
    "points",
    "dense_points",
    "num_voxels",
    "dense_num_voxels",
    "cyv_voxels",
    "cyv_num_voxels",
    "cyv_coordinates",
    "cyv_num_points",
    "gt_boxes_and_cls",
    "reconstruction_coordinates",
    "reconstruction_voxels",
    "reconstruction_num_voxels",
    "reconstruction_num_points",
    "reconstruction_coordinates_4",
    "reconstruction_voxels_4",
    "reconstruction_num_voxels_4",
    "reconstruction_num_points_4",
    "reconstruction_coordinates_2",
    "reconstruction_voxels_2",
    "reconstruction_num_voxels_2",
    "reconstruction_num_points_2",
}


class Voxelization:
    def __init__(
        self,
        range: tuple[float, ...],
        voxel_size: tuple[float, ...],
        max_points_in_voxel: int = 16,
        max_voxel_num: int = 150000,
        **kwargs,
    ):
        self.range = range
        self.voxel_size = voxel_size
        self.max_points_in_voxel = max_points_in_voxel
        self.max_voxel_num = [max_voxel_num, max_voxel_num]
        self.double_flip = False

        generator_kwargs = dict(
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            **generator_kwargs,
        )
        self.voxel_generator_ = VoxelGenerator(
            voxel_size=list(self.voxel_size),
            **generator_kwargs,
        )
        self.voxel_generator_2 = VoxelGenerator(
            voxel_size=[value * 2 for value in self.voxel_size],
            **generator_kwargs,
        )
        self.voxel_generator_4 = VoxelGenerator(
            voxel_size=[value * 4 for value in self.voxel_size],
            **generator_kwargs,
        )

    @staticmethod
    def _voxelize(generator, points, max_voxels, shape, pc_range, voxel_size):
        voxels, coordinates, num_points = generator.generate(
            points, max_voxels=max_voxels
        )
        return dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=np.array([voxels.shape[0]], dtype=np.int64),
            shape=shape,
            range=pc_range,
            size=voxel_size,
        )

    @staticmethod
    def _format_bundle(bundle):
        formatted = {}
        for key, value in bundle.items():
            if key in _TENSOR_KEYS:
                formatted[key] = torch.tensor(value)
            elif key in _COORDINATE_KEYS:
                padded = np.pad(
                    value,
                    ((0, 0), (1, 0)),
                    mode="constant",
                    constant_values=0,
                )
                formatted[key] = torch.tensor(padded)
            else:
                formatted[key] = value
        return formatted

    @staticmethod
    def _move_to_device(example):
        output = {}
        for key, value in example.items():
            if key in _CUDA_KEYS:
                output[key] = value.to("cuda", non_blocking=False)
            elif key == "shape":
                output[key] = value.reshape(1, -1)
            else:
                output[key] = value
        return output

    def __call__(self, points):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size
        max_voxels = self.max_voxel_num[0]

        result = {
            "lidar": {
                "type": "lidar",
                "points": points,
                "annotations": None,
                "dense_points": points,
                "dense_voxels": None,
                "reconstruction_points": points,
                "reconstruction_voxels": None,
                "reconstruction_voxels_2": None,
                "reconstruction_voxels_4": None,
            },
            "metadata": {"num_point_features": 3},
            "calib": None,
            "cam": {},
            "mode": "train",
            "type": "WaymoDataset",
        }

        lidar = result["lidar"]
        lidar["voxels"] = self._voxelize(
            self.voxel_generator,
            lidar["points"],
            max_voxels,
            grid_size,
            pc_range,
            voxel_size,
        )
        lidar["dense_voxels"] = self._voxelize(
            self.voxel_generator,
            lidar["dense_points"],
            max_voxels,
            grid_size,
            pc_range,
            voxel_size,
        )

        reconstruction_points = lidar["reconstruction_points"].astype("float32")
        lidar["reconstruction_voxels"] = self._voxelize(
            self.voxel_generator_,
            reconstruction_points,
            max_voxels,
            grid_size,
            pc_range,
            voxel_size,
        )
        lidar["reconstruction_voxels_2"] = self._voxelize(
            self.voxel_generator_2,
            reconstruction_points,
            max_voxels,
            grid_size,
            pc_range,
            voxel_size,
        )
        lidar["reconstruction_voxels_4"] = self._voxelize(
            self.voxel_generator_4,
            reconstruction_points,
            max_voxels,
            grid_size,
            pc_range,
            voxel_size,
        )

        bundle = Reformat(double_flip=False)(result)
        return self._move_to_device(self._format_bundle(bundle))
