import torch
import numpy as np
import spconv.pytorch as spconv

import torch.nn.functional as F
from torch import nn


class KD_VoxelNet(nn.Module):
    def __init__(
        self,
        reader: nn.Module,
        backbone: nn.Module,
        neck: nn.Module,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        with_neck: bool = False,
        offset_activation: str = "legacy_sigmoid",
    ):
        super().__init__()
        self.reader = reader
        self.backbone = backbone
        self.neck = neck
        self.with_neck = with_neck
        if offset_activation not in {"legacy_sigmoid", "centered_tanh"}:
            raise ValueError(f"Unknown offset activation: {offset_activation}")
        self.offset_activation = offset_activation

    def extract_feat(self, data, train_pcm=True):
        """Encode voxel features with the sparse 3D backbone."""
        input_features = self.reader(data["features"], data["num_voxels"])
        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        if self.with_neck:
            x, gen_offset_2, gen_mask_2, gen_offset_4, gen_mask_4, F_S_a, F_S_b = self.neck(x)
            return (
                x,
                gen_offset_2,
                gen_mask_2,
                gen_offset_4,
                gen_mask_4,
                F_S_a,
                F_S_b,
                voxel_feature,
            )
        return x

    def mask_offset_loss(self, gen_offset, gen_mask, gt, grid):
        """Compute voxel-occupancy BCE and occupied-voxel offset L1 losses."""
        # A voxel is occupied when any XYZ component is present. Using a plain
        # signed sum can incorrectly cancel to zero (for example x + y + z = 0).
        gt_mask = gt[:, :3].abs().sum(1) != 0
        gen_softmax = F.softmax(gen_mask, dim=1)
        gen_mask_new = gen_softmax[:, 0, :, :, :]
        gen_mask_use = (gen_mask_new >= 0.5).float()
        with torch.cuda.amp.autocast(enabled=False):
            N1, D1, H1, W1 = gen_mask_new.shape
            gen_mask_caculate = gen_mask_new.view(N1, D1 * H1 * W1)
            N2, D2, H2, W2 = gt_mask.shape
            gt_mask_caculate = gt_mask.view(N2, D2 * H2 * W2)
            loss = F.binary_cross_entropy(gen_mask_caculate.float(), gt_mask_caculate.float())

        target_offset = gt[:, :3] - grid
        N1, _, D1, H1, W1 = target_offset.shape

        if self.offset_activation == "legacy_sigmoid":
            # Preserve the released/paper implementation for checkpoint
            # compatibility. This branch cannot represent negative offsets.
            length = 51.2 / W1
            gen_offset_use = torch.sigmoid(gen_offset) * length
            offset_mask = target_offset != 0
        else:
            # ``grid`` stores voxel centers, therefore point-minus-center is a
            # signed quantity. Bound each axis to half of its own cell size.
            if D1 < 2 or H1 < 2 or W1 < 2:
                raise ValueError("centered_tanh requires at least two cells per axis")
            dx = (grid[:, 0, :, :, 1:] - grid[:, 0, :, :, :-1]).abs().mean()
            dy = (grid[:, 1, :, 1:, :] - grid[:, 1, :, :-1, :]).abs().mean()
            dz = (grid[:, 2, 1:, :, :] - grid[:, 2, :-1, :, :]).abs().mean()
            half_cell = 0.5 * torch.stack((dx, dy, dz)).view(1, 3, 1, 1, 1)
            gen_offset_use = torch.tanh(gen_offset) * half_cell.to(gen_offset)
            # Offset supervision applies to all three coordinates of occupied
            # voxels, including components whose exact target happens to be 0.
            offset_mask = gt_mask[:, None].expand_as(target_offset)

        if offset_mask.any():
            com_loss = F.l1_loss(
                gen_offset_use[offset_mask], target_offset[offset_mask]
            )
        else:
            com_loss = gen_offset_use.sum() * 0.0
        return loss, com_loss, gen_mask_use, gen_offset_use

    def forward(self, example, return_loss=True, return_feature=False, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        if return_loss:
            reconstruction_voxels = example["reconstruction_voxels_2"]
            reconstruction_coordinates = example["reconstruction_coordinates_2"]
            reconstruction_num_voxels = example["reconstruction_num_voxels_2"]
            reconstruction_num_points_in_voxel = example["reconstruction_num_points_2"]
            sparse_shape = np.array(example["shape"][0][::-1] / 2).astype("int64")
            coors = reconstruction_coordinates.int()
            input_feature = self.reader(
                reconstruction_voxels, reconstruction_num_points_in_voxel
            )
            reconstruction_gt = spconv.SparseConvTensor(
                input_feature,
                coors,
                sparse_shape,
                len(reconstruction_num_voxels),
            ).dense()

            reconstruction_voxels = example["reconstruction_voxels_4"]
            reconstruction_coordinates = example["reconstruction_coordinates_4"]
            reconstruction_num_voxels = example["reconstruction_num_voxels_4"]
            reconstruction_num_points_in_voxel = example["reconstruction_num_points_4"]
            sparse_shape = np.array(example["shape"][0][::-1] / 4).astype("int64")
            coors = reconstruction_coordinates.int()
            input_feature = self.reader(
                reconstruction_voxels, reconstruction_num_points_in_voxel
            )
            reconstruction_gt_4 = spconv.SparseConvTensor(
                input_feature,
                coors,
                sparse_shape,
                len(reconstruction_num_voxels),
            ).dense()

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=len(num_voxels),
            input_shape=example["shape"][0],
        )

        if self.with_neck:
            (
                x,
                gen_offset_2,
                gen_mask_2,
                gen_offset_4,
                gen_mask_4,
                F_S_a,
                F_S_b,
                voxel_feature,
            ) = self.extract_feat(data)
        else:
            x = self.extract_feat(data)

        if return_loss:
            n, _, depth, height, width = gen_offset_2.shape
            zs, ys, xs = torch.meshgrid(
                [
                    torch.arange(0, depth),
                    torch.arange(0, height),
                    torch.arange(0, width),
                ]
            )
            ys = ys * (102.4 / height) - 51.2 + (102.4 / height) / 2
            xs = xs * (102.4 / width) - 51.2 + (102.4 / height) / 2
            zs = zs * (6 / depth) - 2 + (6 / depth) / 2
            grid = (
                torch.cat([xs[None], ys[None], zs[None]], 0)[None]
                .repeat(n, 1, 1, 1, 1)
                .to(reconstruction_gt)
            )
            mask_loss_2, offset_loss_2, _, _ = self.mask_offset_loss(
                gen_offset_2, gen_mask_2, reconstruction_gt, grid
            )

            n, _, depth, height, width = reconstruction_gt_4.shape
            zs, ys, xs = torch.meshgrid(
                [
                    torch.arange(0, depth),
                    torch.arange(0, height),
                    torch.arange(0, width),
                ]
            )
            ys = ys * (102.4 / height) - 51.2 + (102.4 / height) / 2
            xs = xs * (102.4 / width) - 51.2 + (102.4 / height) / 2
            zs = zs * (6 / depth) - 2 + (6 / depth) / 2
            grid_4 = (
                torch.cat([xs[None], ys[None], zs[None]], 0)[None]
                .repeat(n, 1, 1, 1, 1)
                .to(reconstruction_gt_4)
            )
            mask_loss_4, offset_loss_4, _, _ = self.mask_offset_loss(
                gen_offset_4, gen_mask_4, reconstruction_gt_4, grid_4
            )

            mask_loss = mask_loss_2 + mask_loss_4
            comp_loss = offset_loss_2 + offset_loss_4
            return F_S_a, F_S_b, mask_loss, comp_loss

        return x

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, _, _, _, _, F_S_a, F_S_b, voxel_feature = self.extract_feat(
            data, train_pcm=False
        )

        bev_feature = x
        preds = self.bbox_head(x)

        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, voxel_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, voxel_feature, None, F_S_a, F_S_b
