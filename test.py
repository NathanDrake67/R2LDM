"""Generate dense point clouds with a trained R2LDM checkpoint."""

import argparse
from pathlib import Path

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time

import utils.option
from utils.inference import count_parameters
from utils.lidar import get_hdl64e_linear_ray_angles

warnings.filterwarnings("ignore", category=UserWarning)


def load_data_as_points(file_path: str) -> np.ndarray:
    """Load XYZ values from a preprocessed VoD LiDAR or radar binary."""
    if "lidar" in file_path:
        points_file = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
        points_array = points_file[:, :3]
    elif "radar" in file_path:
        points_file = np.fromfile(file_path, dtype=np.float32).reshape((-1, 7))
        points_array = points_file[:, :3]
    else:
        raise ValueError(f"Cannot infer point-cloud format from path: {file_path}")

    return points_array


def run_inference(
    cfg: utils.option.Config,
    checkpoint_path: str,
    max_samples: int | None = None,
):
    # Delay sparse-CUDA imports so ``--help`` remains available before the
    # environment-specific spconv dependency is installed.
    from det3d.datasets.vod.vod_old import VodDataset
    from models.diffusion import (
        ContinuousTimeGaussianDiffusion,
        DiscreteTimeGaussianDiffusion,
    )
    from models.efficient_unet import EfficientUNet
    from models.refinenet import LiDARGenRefineNet
    from utils.voxel import (
        build_lpcr_encoder_pair,
        build_voxelizer,
        validate_checkpoint_voxel_geometry,
    )

    if cfg.training.batch_size_eval != 1:
        raise ValueError(
            "R2LDM inference currently requires "
            "training.batch_size_eval=1; got "
            f"{cfg.training.batch_size_eval}."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("R2LDM inference requires a CUDA-capable GPU.")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)

    print(cfg)
    device = torch.device("cuda")

    # =================================================================================
    # Setup models
    # =================================================================================

    channels = [
        1 if cfg.data.train_depth else 0,
        1 if cfg.data.train_reflectance else 0,
    ]

    voxelizer = build_voxelizer(cfg.voxelextractor)
    point_cloud_bounds = cfg.voxelextractor.grid_bounds()
    kd_voxelnet, kd_voxelnet_r = build_lpcr_encoder_pair(
        cfg.voxelextractor, device
    )

    if cfg.model.architecture == "efficient_unet":
        model = EfficientUNet(
            model=kd_voxelnet,
            model_r=kd_voxelnet_r,
            in_channels=sum(channels),
            resolution=cfg.data.resolution,
            base_channels=cfg.model.base_channels,
            temb_channels=cfg.model.temb_channels,
            channel_multiplier=cfg.model.channel_multiplier,
            num_residual_blocks=cfg.model.num_residual_blocks,
            gn_num_groups=cfg.model.gn_num_groups,
            gn_eps=cfg.model.gn_eps,
            attn_num_heads=cfg.model.attn_num_heads,
            coords_encoding=cfg.model.coords_encoding,
            ring=True,
        )
    elif cfg.model.architecture == "refinenet":
        model = LiDARGenRefineNet(
            in_channels=sum(channels),
            resolution=cfg.data.resolution,
            base_channels=cfg.model.base_channels,
            channel_multiplier=cfg.model.channel_multiplier,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model.architecture}")

    if "spherical" in cfg.data.projection:
        model.coords = get_hdl64e_linear_ray_angles(*cfg.data.resolution)
    elif "unfolding" in cfg.data.projection:
        model.coords = F.interpolate(
            torch.load(f"data/{cfg.data.dataset}/unfolding_angles.pth"),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
    else:
        raise ValueError(f"Unknown: {cfg.data.projection}")

    if cfg.diffusion.timestep_type == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            model=model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
            num_training_steps=cfg.diffusion.num_training_steps,
        )
    elif cfg.diffusion.timestep_type == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            model=model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")

    ddpm.to(device)

    print(f"number of parameters: {count_parameters(ddpm):,}")


    # =================================================================================
    # Setup dataset and dataloader
    # =================================================================================

    data_root = Path(cfg.data.root).expanduser().resolve()
    test_split = Path(cfg.data.test_split).expanduser().resolve()
    output_root = Path(cfg.data.prediction_dir).expanduser().resolve()
    for required_path in (data_root, test_split):
        if not required_path.exists():
            raise FileNotFoundError(f"Required data path does not exist: {required_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    test_data = VodDataset(
        root_path=str(data_root),
        pipeline=None,
        filenames=str(test_split),
        lidar_input=cfg.data.lidar_input,
        lidar_gt=cfg.data.lidar_gt,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=cfg.training.batch_size_eval,
        num_workers=cfg.training.num_workers,
    )


    # =================================================================================
    checkpoint_path_obj = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path_obj.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path_obj}")
    checkpoint = torch.load(checkpoint_path_obj, map_location="cpu")
    saved_cfg = checkpoint.get("cfg", {})
    saved_voxel_cfg = (
        saved_cfg.get("voxelextractor", {}) if isinstance(saved_cfg, dict) else {}
    )
    saved_offset_activation = saved_voxel_cfg.get(
        "offset_activation", "legacy_sigmoid"
    )
    if saved_offset_activation != cfg.voxelextractor.offset_activation:
        raise ValueError(
            "Checkpoint LPCR offset mode is "
            f"{saved_offset_activation!r}, but inference uses "
            f"{cfg.voxelextractor.offset_activation!r}. Pass the matching "
            "--offset-activation value."
        )
    validate_checkpoint_voxel_geometry(saved_voxel_cfg, cfg.voxelextractor)
    state_dict = (
        checkpoint["ema_weights"]
        if "ema_weights" in checkpoint
        else checkpoint["weights"]
    )
    ddpm.load_state_dict(state_dict)
    ddpm.eval()
    ddpm.to(device)
    for parameter in ddpm.parameters():
        parameter.requires_grad = False
    print(
        f"loaded checkpoint {checkpoint_path_obj} "
        f"(stage={checkpoint.get('train_stage')}, epoch={checkpoint.get('epoch')})"
    )

    # =================================================================================
    # Inference loop
    # =================================================================================
    total_time = 0.0
    processed = 0
    progress_bar = tqdm(test_loader, desc="inference", dynamic_ncols=True)
    with torch.inference_mode():
        for data in progress_bar:
            lidar_path, radar_path, _ = data
            lidar_path = Path(str(lidar_path[0])).resolve()
            radar_path = Path(str(radar_path[0])).resolve()
            lidar_points = load_data_as_points(str(lidar_path))
            radar_points = load_data_as_points(str(radar_path))

            radar_example = voxelizer(radar_points)
            lidar_example = voxelizer(lidar_points)

            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            _, gen_points = ddpm(
                lidar=lidar_example,
                radar=radar_example,
                lidar_ds=lidar_example,
                grid_size=point_cloud_bounds,
                stage=3,
                global_step=processed,
                num_sampling_steps=cfg.diffusion.num_sampling_steps,
            )
            torch.cuda.synchronize(device)
            total_time += time.perf_counter() - start_time

            relative_path = radar_path.relative_to(data_root)
            relative_parts = list(relative_path.parts)
            try:
                relative_parts.remove("radar")
            except ValueError as exc:
                raise ValueError(
                    f"Expected a 'radar' directory in dataset path: {radar_path}"
                ) from exc
            output_path = output_root.joinpath(*relative_parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            gen_tensor = (
                gen_points[0]
                .transpose(0, 1)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            gen_tensor.tofile(output_path)
            processed += 1

            if max_samples is not None and processed >= max_samples:
                break

    average_time = total_time / processed if processed else 0.0
    timing_log = output_root / "timing.txt"
    timing_log.write_text(
        f"samples: {processed}\n"
        f"total_seconds: {total_time:.4f}\n"
        f"average_seconds: {average_time:.4f}\n",
        encoding="utf-8",
    )
    print(f"wrote {processed} predictions to {output_root}")
    print(f"average inference time: {average_time:.4f} s/sample")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate dense point clouds with a trained R2LDM checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Stage-2 R2LDM checkpoint")
    parser.add_argument("--data-root", help="Root of the preprocessed VoD dataset")
    parser.add_argument("--test-split", help="Test split text file")
    parser.add_argument("--output-dir", help="Directory for generated .bin files")
    parser.add_argument("--sampling-steps", type=int, help="DDIM sampling steps")
    parser.add_argument("--max-samples", type=int, help="Optional smoke-test limit")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--offset-activation",
        choices=("legacy_sigmoid", "centered_tanh"),
        help="Must match the LPCR mode used to train the checkpoint",
    )
    utils.option.add_voxel_config_arguments(parser)
    return parser


def config_from_args(args: argparse.Namespace) -> utils.option.Config:
    cfg = utils.option.Config()
    if args.data_root is not None:
        cfg.data.root = args.data_root
    if args.test_split is not None:
        cfg.data.test_split = args.test_split
    if args.output_dir is not None:
        cfg.data.prediction_dir = args.output_dir
    if args.sampling_steps is not None:
        cfg.diffusion.num_sampling_steps = args.sampling_steps
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.offset_activation is not None:
        cfg.voxelextractor.offset_activation = args.offset_activation
    utils.option.apply_voxel_config_overrides(cfg, args)
    return cfg


def main() -> None:
    args = build_parser().parse_args()
    run_inference(
        config_from_args(args),
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
