from typing import List, Literal

import einops
import torch
from torch import nn
from torch.cuda.amp import autocast
import utils.inference
import numpy as np
import spconv.pytorch as spconv
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    """
    Base class for continuous/discrete Gaussian diffusion models
    """

    def __init__(
        self,
        model: nn.Module,
        sampling: Literal["ddpm", "ddim"] = "ddim",
        prediction_type: Literal["eps", "v", "x_0"] = "eps",
        loss_type: Literal["l2", "l1", "huber"] | nn.Module = "l2",
        num_training_steps: int = 1000,
        noise_schedule: Literal["linear", "cosine", "sigmoid"] = "linear",
        min_snr_loss_weight: bool = True,
        min_snr_gamma: float = 5.0,
        sampling_resolution: tuple[int, int] | None = None,
        clip_sample: bool = True,
        clip_sample_range: float = 1,
    ):
        super().__init__()
        self.model = model
        self.sampling = sampling
        self.num_training_steps = num_training_steps
        self.objective = prediction_type
        self.noise_schedule = noise_schedule
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        if loss_type == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif isinstance(loss_type, nn.Module):
            self.criterion = loss_type
        else:
            raise ValueError(f"invalid criterion: {loss_type}")
        if hasattr(self.criterion, "reduction"):
            assert self.criterion.reduction == "none"

        if sampling_resolution is None:
            assert hasattr(self.model, "resolution")
            assert hasattr(self.model, "in_channels")
            self.sampling_shape = (
                self.model.in_channels,
                *self.model.resolution,
            )
        else:
            assert len(sampling_resolution) == 2
            assert hasattr(self.model, "in_channels")
            self.sampling_shape = (self.model.in_channels, *sampling_resolution)

        self.setup_parameters()
        self.register_buffer("_dummy", torch.tensor([]))

    @property
    def device(self):
        return self._dummy.device

    def randn(
        self,
        *shape,
        rng: List[torch.Generator] | torch.Generator | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if rng is None:
            return torch.randn(*shape, **kwargs)
        elif isinstance(rng, torch.Generator):
            return torch.randn(*shape, generator=rng, **kwargs)
        elif isinstance(rng, list):
            assert len(rng) == shape[0]
            return torch.stack(
                [torch.randn(*shape[1:], generator=r, **kwargs) for r in rng]
            )
        else:
            raise ValueError(f"invalid rng: {rng}")

    def randn_like(
        self,
        x: torch.Tensor,
        rng: List[torch.Generator] | torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.randn(*x.shape, rng=rng, device=x.device, dtype=x.dtype)

    def setup_parameters(self) -> None:
        raise NotImplementedError

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def get_network_condition(self, steps: torch.Tensor, cond: torch.Tensor):
        raise NotImplementedError

    def get_target(self, x_0, steps, noise):
        raise NotImplementedError

    def get_loss_weight(self, steps):
        raise NotImplementedError

    @autocast(enabled=False)
    def q_step_from_x_0(self, x_0, steps, rng):
        raise NotImplementedError

    def q_step(self, *args, **kwargs):
        raise NotImplementedError

    @torch.inference_mode()
    def p_step(self, *args, **kwargs):
        raise NotImplementedError

    def p_loss(
        self,
        x_0: torch.Tensor,
        steps: torch.Tensor,
        cond: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Shared by the continuous- and discrete-time diffusion models.
        loss_mask = torch.ones_like(x_0) if loss_mask is None else loss_mask
        x_t, noise = self.q_step_from_x_0(x_0, steps)
        time_condition = self.get_network_condition(steps, cond)
        prediction = self.model(x_t, time_condition, cond)
        target = self.get_target(x_0, steps, noise)
        loss = self.criterion(prediction, target)
        loss = einops.reduce(loss * loss_mask, "B ... -> B ()", "sum")
        loss_mask = einops.reduce(loss_mask, "B ... -> B ()", "sum")
        loss = loss / loss_mask.add(1e-8)
        loss = (loss * self.get_loss_weight(steps)).mean()
        return loss

    def forward(
        self,
        lidar,  
        radar,
        lidar_ds,
        grid_size,
        stage: int = 1,
        global_step: int = 0,
        loss_mask: torch.Tensor | None = None,
        num_sampling_steps: int = 128,
    ) -> torch.Tensor:
        if stage == 0:
            lidar_ds_0_old = self.model.preprocess_r(lidar_ds, return_loss=False)
            lidar_ds_0_new = F.tanh(lidar_ds_0_old)
            lidar_ds_0 = lidar_ds_0_new.to("cuda")
            x, gen_offset_2, gen_mask_2, gen_offset_1, gen_mask_1, F_S_a, F_S_b = (
                self.model.preprocess_r.neck(lidar_ds_0)
            )

            reconstruction_voxels = lidar["voxels"]
            reconstruction_coordinates = lidar["coordinates"]
            reconstruction_num_voxels = lidar["num_voxels"]
            reconstruction_num_points_in_voxel = lidar["num_points"]
            sparse_shape = np.array(lidar["shape"][0][::-1] / 1).astype("int64")
            coors = reconstruction_coordinates.int()
            input_feature = self.model.preprocess.reader(
                reconstruction_voxels, reconstruction_num_points_in_voxel
            )
            reconstruction_gt = spconv.SparseConvTensor(
                input_feature,
                coors,
                sparse_shape,
                len(reconstruction_num_voxels),
            ).dense()

            x_former = grid_size[0][0]
            x_latter = grid_size[0][1]
            y_former = grid_size[1][0]
            y_latter = grid_size[1][1]
            z_former = grid_size[2][0]
            z_latter = grid_size[2][1]
            N, _, D, H, W = gen_offset_1.shape
            zs, ys, xs = torch.meshgrid(
                [torch.arange(0, D), torch.arange(0, H), torch.arange(0, W)]
            )
            ys = ys * ((y_latter - y_former) / H) + y_former + (
                (y_latter - y_former) / H
            ) / 2
            xs = xs * ((x_latter - x_former) / W) + x_former + (
                (x_latter - x_former) / H
            ) / 2
            zs = zs * ((z_latter - z_former) / D) + z_former + (
                (z_latter - z_former) / D
            ) / 2
            grid = (
                torch.cat([xs[None], ys[None], zs[None]], 0)[None]
                .repeat(N, 1, 1, 1, 1)
                .to(reconstruction_gt)
            )
            mask_loss_1, offset_loss_1, gen_mask_new_1, gen_offset_new_1 = (
                self.model.preprocess_r.mask_offset_loss(
                    gen_offset_1, gen_mask_1, reconstruction_gt, grid
                )
            )
            loss_msk = mask_loss_1
            loss_offst = offset_loss_1
            loss = loss_msk + 0.04 * loss_offst

        if stage == 1:
            x_0 = self.model.preprocess_r(lidar_ds, return_loss=False)
            cond = self.model.preprocess(radar, return_loss=False)
            x_0 = F.tanh(x_0).to("cuda")
            cond = F.tanh(cond).to("cuda")

            steps = self.sample_timesteps(x_0.shape[0], x_0.device)

            normalized_x0 = x_0.clamp(0, 1)
            normalized_cond = cond.clamp(0, 1)
            
            from utils.lidar import LiDARUtility
            x_0_in = LiDARUtility.normalize(normalized_x0)
            cond_in = LiDARUtility.normalize(normalized_cond)

            loss = self.p_loss(x_0_in, steps, cond_in, loss_mask)

        if stage == 3:
            x_0 = self.model.preprocess_r(lidar_ds, return_loss=False)
            cond = self.model.preprocess(radar, return_loss=False)
            x_0 = F.tanh(x_0).to("cuda")
            cond = F.tanh(cond).to("cuda")

            x_in = torch.rand(x_0.shape).to("cuda")
            mask = torch.zeros_like(x_0).to("cuda")

            normalized_x0 = x_0.clamp(0, 1)
            normalized_cond = cond.clamp(0, 1)
            from utils.lidar import LiDARUtility

            x_0_in = LiDARUtility.normalize(normalized_x0)
            cond_in = LiDARUtility.normalize(normalized_cond)
            x_in_in = LiDARUtility.normalize(x_in)

            x_out = self.cond_generate(
                known=x_in_in,
                condition=cond_in,
                mask=mask,
                num_steps=num_sampling_steps,
                num_resample_steps=4,
                jump_length=1,
                rng=utils.inference.setup_rng(range(1), device="cuda"),
            ).clamp(-1, 1)

            x_out_norm = LiDARUtility.denormalize(x_out)
            x_out_final = x_out_norm

            x, gen_offset_2, gen_mask_2, gen_offset_1, gen_mask_1, F_S_a, F_S_b = (
                self.model.preprocess_r.neck(x_out_final)
            )
            reconstruction_voxels = lidar["voxels"]
            reconstruction_coordinates = lidar["coordinates"]
            reconstruction_num_voxels = lidar["num_voxels"]
            reconstruction_num_points_in_voxel = lidar["num_points"]
            sparse_shape = np.array(lidar["shape"][0][::-1] / 1).astype("int64")
            coors = reconstruction_coordinates.int()
            input_feature = self.model.preprocess.reader(
                reconstruction_voxels, reconstruction_num_points_in_voxel
            )
            reconstruction_gt = spconv.SparseConvTensor(
                input_feature,
                coors,
                sparse_shape,
                len(reconstruction_num_voxels),
            ).dense()
            x_former = grid_size[0][0]
            x_latter = grid_size[0][1]
            y_former = grid_size[1][0]
            y_latter = grid_size[1][1]
            z_former = grid_size[2][0]
            z_latter = grid_size[2][1]

            N, _, D, H, W = gen_offset_1.shape
            zs, ys, xs = torch.meshgrid(
                [torch.arange(0, D), torch.arange(0, H), torch.arange(0, W)]
            )
            ys = ys * ((y_latter - y_former) / H) + y_former + (
                (y_latter - y_former) / H
            ) / 2
            xs = xs * ((x_latter - x_former) / W) + x_former + (
                (x_latter - x_former) / H
            ) / 2
            zs = zs * ((z_latter - z_former) / D) + z_former + (
                (z_latter - z_former) / D
            ) / 2
            grid = (
                torch.cat([xs[None], ys[None], zs[None]], 0)[None]
                .repeat(N, 1, 1, 1, 1)
                .to(reconstruction_gt)
            )
            mask_loss_1, offset_loss_1, gen_mask_new_1, gen_offset_new_1 = (
                self.model.preprocess_r.mask_offset_loss(
                    gen_offset_1, gen_mask_1, reconstruction_gt, grid
                )
            )

            points = grid * gen_mask_new_1 + gen_offset_new_1 * gen_mask_new_1
            N, C, D, H, W = points.shape
            points = points.view(N * C, D * H * W).cpu()
            transposed_points = points.permute(1, 0)

            gt_points = reconstruction_gt.view(N * C, D * H * W).cpu()
            gt_transposed_points = gt_points.permute(1, 0)
            loss = mask_loss_1 + 0.04 * offset_loss_1

            is_zero_column = (points == 0).all(dim=0)
            non_zero_columns = points[:, ~is_zero_column]
            transposed_points = non_zero_columns.unsqueeze(0)
            is_zero_column_2 = (gt_points == 0).all(dim=0)
            non_zero_columns_2 = gt_points[:, ~is_zero_column_2]
            gt_transposed_points = non_zero_columns_2.unsqueeze(0)

        if stage == 3:
            return gt_transposed_points, transposed_points

        else:
            return loss
