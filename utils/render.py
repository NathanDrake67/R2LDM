"""Point-cloud rendering and visualization utilities."""

import kornia
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.conversions import axis_angle_to_rotation_matrix


def make_Rt(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    device: torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create batched rotation and translation tensors."""
    zero = torch.zeros(1, device=device)
    roll = torch.full_like(zero, fill_value=roll, device=device)
    pitch = torch.full_like(zero, fill_value=pitch, device=device)
    yaw = torch.full_like(zero, fill_value=yaw, device=device)

    rotation = axis_angle_to_rotation_matrix(
        torch.stack([zero, zero, yaw], dim=-1)
    )
    rotation @= axis_angle_to_rotation_matrix(
        torch.stack([zero, pitch, zero], dim=-1)
    )
    rotation @= axis_angle_to_rotation_matrix(
        torch.stack([roll, zero, zero], dim=-1)
    )
    translation = torch.tensor([[x, y, z]], device=device)
    return rotation, translation


def render_point_clouds(
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    size: int = 800,
    R: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    focal_length=1.0,
) -> torch.Tensor:
    """Render batched point clouds with a differentiable rasterizer."""
    points = points.clone()
    points[..., 2] *= -1
    device = points.device

    if colors is None:
        batch_size, num_points, _ = points.shape
        colors = torch.ones(batch_size, num_points, 3).to(points)

    if R is not None:
        assert R.shape[-2:] == (3, 3)
        points = points @ R
    if t is not None:
        assert t.shape[-1:] == (3,)
        points += t

    intrinsics = torch.eye(3, device=device)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length
    intrinsics[0, 2] = 0.5
    intrinsics[1, 2] = 0.5
    intrinsics = intrinsics[None]

    uv = kornia.geometry.project_points(points, intrinsics) * size
    mask = (0 < uv) & (uv < size - 1)
    mask = torch.logical_and(mask[..., [0]], mask[..., [1]])
    colors = colors * mask

    uv = size - uv
    depth = torch.norm(points, p=2, dim=-1, keepdim=True)
    weight = 1.0 / torch.exp(3.0 * depth)
    weight *= (depth > 1e-8).detach()
    bev = bilinear_rasterizer(uv, weight * colors, (size, size))
    bev /= bilinear_rasterizer(uv, weight, (size, size)) + 1e-8
    return bev


def bilinear_rasterizer(
    coords: torch.Tensor,
    values: torch.Tensor,
    out_shape: tuple[int, int],
) -> torch.Tensor:
    """Rasterize point values with bilinear interpolation.

    Adapted from VCL3D/SphericalViewSynthesis.
    """
    batch_size, _, channels = values.shape
    height, width = out_shape
    device = coords.device

    h = coords[..., [0]].expand(-1, -1, channels)
    w = coords[..., [1]].expand(-1, -1, channels)

    h_top = torch.floor(h)
    h_bottom = h_top + 1
    w_left = torch.floor(w)
    w_right = w_left + 1

    h_top_safe = torch.clamp(h_top, 0.0, height - 1)
    h_bottom_safe = torch.clamp(h_bottom, 0.0, height - 1)
    w_left_safe = torch.clamp(w_left, 0.0, width - 1)
    w_right_safe = torch.clamp(w_right, 0.0, width - 1)

    weight_h_top = (h_bottom - h) * (h_top == h_top_safe).detach().float()
    weight_h_bottom = (h - h_top) * (
        h_bottom == h_bottom_safe
    ).detach().float()
    weight_w_left = (w_right - w) * (w_left == w_left_safe).detach().float()
    weight_w_right = (w - w_left) * (
        w_right == w_right_safe
    ).detach().float()

    weight_top_left = weight_h_top * weight_w_left
    weight_top_right = weight_h_top * weight_w_right
    weight_bottom_left = weight_h_bottom * weight_w_left
    weight_bottom_right = weight_h_bottom * weight_w_right

    weight_top_left *= (weight_top_left >= 1e-3).detach().float()
    weight_top_right *= (weight_top_right >= 1e-3).detach().float()
    weight_bottom_left *= (weight_bottom_left >= 1e-3).detach().float()
    weight_bottom_right *= (weight_bottom_right >= 1e-3).detach().float()

    values_top_left = values * weight_top_left
    values_top_right = values * weight_top_right
    values_bottom_left = values * weight_bottom_left
    values_bottom_right = values * weight_bottom_right

    indices_top_left = (w_left_safe + width * h_top_safe).long()
    indices_top_right = (w_right_safe + width * h_top_safe).long()
    indices_bottom_left = (w_left_safe + width * h_bottom_safe).long()
    indices_bottom_right = (w_right_safe + width * h_bottom_safe).long()

    render = torch.zeros(
        batch_size, height * width, channels, device=device
    )
    render.scatter_add_(1, indices_top_left, values_top_left)
    render.scatter_add_(1, indices_top_right, values_top_right)
    render.scatter_add_(1, indices_bottom_left, values_bottom_left)
    render.scatter_add_(1, indices_bottom_right, values_bottom_right)
    return render.reshape(batch_size, height, width, channels).permute(
        0, 3, 1, 2
    )


def estimate_surface_normal(
    points: torch.Tensor,
    d: int = 2,
    mode: str = "closest",
) -> torch.Tensor:
    """Estimate normals from an organized point cloud."""
    assert points.dim() == 4, f"expected (B,3,H,W), but got {points.shape}"
    batch_size, channels, height, width = points.shape
    assert channels == 3, f"expected C==3, but got {channels}"
    device = points.device

    points = F.pad(points, (0, 0, d, d), mode="replicate")
    points = F.pad(points, (d, d, 0, 0), mode="circular")
    points = points.permute(0, 2, 3, 1)

    offsets = torch.tensor(
        [
            (-d, 0),
            (-d, d),
            (0, d),
            (d, d),
            (d, 0),
            (d, -d),
            (0, -d),
            (-d, -d),
        ],
        device=device,
    )

    batch = torch.arange(batch_size, device=device)[:, None, None]
    rows = torch.arange(height, device=device)[None, :, None]
    columns = torch.arange(width, device=device)[None, None, :]
    neighbors = torch.arange(8, device=device)

    batch_anchor = batch[:, None]
    row_anchor = rows[:, None] + d
    column_anchor = columns[:, None] + d
    anchors = points[batch_anchor, row_anchor, column_anchor]

    offset = offsets[neighbors]
    rows_1 = row_anchor + offset[None, :, 0, None, None]
    columns_1 = column_anchor + offset[None, :, 1, None, None]
    points_1 = points[batch_anchor, rows_1, columns_1]

    offset = offsets[(neighbors + 2) % 8]
    rows_2 = row_anchor + offset[None, :, 0, None, None]
    columns_2 = column_anchor + offset[None, :, 1, None, None]
    points_2 = points[batch_anchor, rows_2, columns_2]

    if mode == "closest":
        difference = torch.norm(points_1 - anchors, dim=4)
        difference += torch.norm(points_2 - anchors, dim=4)
        closest = torch.argmin(difference, dim=1)
        anchors = anchors[batch, 0, rows, columns]
        points_1 = points_1[batch, closest, rows, columns]
        points_2 = points_2[batch, closest, rows, columns]
        normals = torch.cross(points_1 - anchors, points_2 - anchors, dim=-1)
    elif mode == "mean":
        normals = torch.cross(points_1 - anchors, points_2 - anchors, dim=-1)
        normals = normals.mean(dim=1)
    else:
        raise NotImplementedError(mode)

    normals = normals / (torch.norm(normals, dim=3, keepdim=True) + 1e-8)
    return normals.permute(0, 3, 1, 2)


@torch.no_grad()
def colorize(tensor: torch.Tensor, cmap_fn=cm.turbo) -> torch.Tensor:
    """Map normalized scalar values to RGB bytes."""
    colors = cmap_fn(np.linspace(0, 1, 256))[:, :3]
    colors = torch.from_numpy(colors).to(tensor)
    tensor = tensor.squeeze(1) if tensor.ndim == 4 else tensor
    color_ids = (tensor * 256).clamp(0, 255).long()
    tensor = F.embedding(color_ids, colors).permute(0, 3, 1, 2)
    return tensor.mul(255).clamp(0, 255).byte()
