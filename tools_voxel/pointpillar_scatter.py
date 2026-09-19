"""Scatter sparse pillar features into dense BEV feature maps."""

import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, grid_size, **kwargs):
        super().__init__()
        self.num_bev_features = 64
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 4

    def forward(self, pillar_features, coords, **kwargs):
        """Scatter ``(M, C)`` pillar features using their sparse coordinates."""
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_index in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device,
            )
            batch_mask = coords[:, 0] == batch_index
            batch_coords = coords[batch_mask, :]
            indices = (
                batch_coords[:, 0]
                + batch_coords[:, 1] * self.nx
                + batch_coords[:, 2]
            ).long()
            spatial_feature[:, indices] = pillar_features[batch_mask, :].t()
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        return batch_spatial_features.view(
            batch_size,
            self.num_bev_features * self.nz,
            self.ny,
            self.nx,
        )


class PointPillarScatter3d(nn.Module):
    def __init__(self, grid_size, **kwargs):
        super().__init__()
        self.nx, self.ny, self.nz = grid_size
        self.num_bev_features = 64
        self.num_bev_features_before_compression = self.num_bev_features // self.nz

    def forward(self, pillar_features, coords, **kwargs):
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_index in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device,
            )
            batch_mask = coords[:, 0] == batch_index
            batch_coords = coords[batch_mask, :]
            indices = (
                batch_coords[:, 1] * self.ny * self.nx
                + batch_coords[:, 2] * self.nx
                + batch_coords[:, 3]
            ).long()
            spatial_feature[:, indices] = pillar_features[batch_mask, :].t()
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        return batch_spatial_features.view(
            batch_size,
            self.num_bev_features_before_compression * self.nz,
            self.ny,
            self.nx,
        )
