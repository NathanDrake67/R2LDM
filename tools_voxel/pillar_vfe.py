"""Pillar feature encoder used by the legacy PointPillars utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_norm=True,
        last_layer=False,
    ):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            num_parts = inputs.shape[0] // self.part
            linear_outputs = [
                self.linear(inputs[index * self.part : (index + 1) * self.part])
                for index in range(num_parts + 1)
            ]
            x = torch.cat(linear_outputs, dim=0)
        else:
            x = self.linear(inputs)

        torch.backends.cudnn.enabled = False
        if self.use_norm:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max

        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        return torch.cat([x, x_repeat], dim=2)


class PillarVFE(VFETemplate):
    """Encode the points in each pillar with a compact PointNet layer."""

    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg)

        num_point_features = 3
        point_cloud_range = [-32, -32, -1.0, 32, 32, 7.0]
        self.use_norm = True
        self.with_distance = False
        self.use_absolute_xyz = True

        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = [64]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        self.pfn_layers = nn.ModuleList(
            [
                PFNLayer(
                    num_filters[index],
                    num_filters[index + 1],
                    self.use_norm,
                    last_layer=index >= len(num_filters) - 2,
                )
                for index in range(len(num_filters) - 1)
            ]
        )

        voxel_size = [0.16, 0.16, 8]
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(
            max_num, dtype=torch.int, device=actual_num.device
        ).view(max_num_shape)
        return actual_num.int() > max_num

    def forward(self, voxel_features, coords, voxel_num_points, **kwargs):
        points_mean = voxel_features[:, :, :3].sum(
            dim=1, keepdim=True
        ) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x
            + self.x_offset
        )
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y
            + self.y_offset
        )
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 0].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z
            + self.z_offset
        )

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            features.append(
                torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            )

        features = torch.cat(features, dim=-1)
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(
            voxel_num_points, voxel_count, axis=0
        )
        features *= torch.unsqueeze(mask, -1).type_as(voxel_features)

        for pfn in self.pfn_layers:
            features = pfn(features)
        return features.squeeze()
