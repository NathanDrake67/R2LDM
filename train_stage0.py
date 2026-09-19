"""Train either stage of R2LDM using the historical public entry-point name."""

import argparse
import dataclasses
import datetime
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import utils.option
from utils.lidar import get_hdl64e_linear_ray_angles

from torch.utils.tensorboard import SummaryWriter
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

def train(cfg: utils.option.Config, stage_name: str):
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

    if stage_name not in utils.option.STAGE_PRESETS:
        raise ValueError(f"Unknown training stage: {stage_name}")

    preset = utils.option.STAGE_PRESETS[stage_name]
    train_stage = int(preset["stage_id"])
    learning_rate = cfg.training.lr if cfg.training.lr is not None else preset["lr"]
    max_epochs = (
        cfg.training.max_epochs
        if cfg.training.max_epochs is not None
        else int(preset["epochs"])
    )

    if cfg.training.batch_size_train != 1:
        raise ValueError(
            "R2LDM sparse point-cloud preprocessing currently requires "
            "training.batch_size_train=1; got "
            f"{cfg.training.batch_size_train}."
        )
    if cfg.training.batch_size_eval != 1:
        raise ValueError(
            "R2LDM validation currently requires "
            "training.batch_size_eval=1; got "
            f"{cfg.training.batch_size_eval}."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("R2LDM training requires a CUDA-capable GPU.")

    torch.backends.cudnn.benchmark = True
    project_dir = (
        Path(cfg.training.output_dir)
        / cfg.data.dataset
        / cfg.data.projection
        / stage_name
    )

    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)

    print(
        f"training stage: {stage_name} (id={train_stage}), "
        f"lr={learning_rate:g}, epochs={max_epochs}"
    )

    print(cfg)
    os.makedirs(project_dir, exist_ok=True)
    project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    device = torch.device("cuda")
    logging_dir = project_dir / project_name


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

    ddpm.train()
    ddpm.to(device)

    ddpm_ema = EMA(
        ddpm,
        beta=cfg.training.ema_decay,
        update_every=cfg.training.ema_update_every,
        update_after_step=cfg.training.lr_warmup_steps
                          * cfg.training.gradient_accumulation_steps,
    )
    ddpm_ema.to(device)

    # =================================================================================
    # Setup dataloader
    # =================================================================================

    data_root = Path(cfg.data.root).expanduser().resolve()
    train_split = Path(cfg.data.train_split).expanduser().resolve()
    val_split = Path(cfg.data.val_split).expanduser().resolve()
    for required_path in (data_root, train_split, val_split):
        if not required_path.exists():
            raise FileNotFoundError(f"Required data path does not exist: {required_path}")

    train_data = VodDataset(
        root_path=str(data_root),
        pipeline=None,
        filenames=str(train_split),
        lidar_input=cfg.data.lidar_input,
        lidar_gt=cfg.data.lidar_gt,
    )
    val_data = VodDataset(
        root_path=str(data_root),
        pipeline=None,
        filenames=str(val_split),
        lidar_input=cfg.data.lidar_input,
        lidar_gt=cfg.data.lidar_gt,
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.training.batch_size_train,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    valid_loader = DataLoader(
        dataset=val_data,
        batch_size=cfg.training.batch_size_eval,
        num_workers=cfg.training.num_workers,
    )

    # =================================================================================
    # Restore a checkpoint or initialize Stage 2 from the LPCR checkpoint
    # =================================================================================
    checkpoint = None
    start_epoch = 1
    checkpoint_path = cfg.training.resume_checkpoint
    is_resume = checkpoint_path is not None
    if checkpoint_path is None and train_stage == 1:
        checkpoint_path = cfg.training.lpcr_checkpoint

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            purpose = "resume" if is_resume else "LPCR"
            raise FileNotFoundError(
                f"{purpose} checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_cfg = checkpoint.get("cfg", {})
        saved_voxel_cfg = (
            saved_cfg.get("voxelextractor", {})
            if isinstance(saved_cfg, dict)
            else {}
        )
        saved_offset_activation = saved_voxel_cfg.get(
            "offset_activation", "legacy_sigmoid"
        )
        if saved_offset_activation != cfg.voxelextractor.offset_activation:
            raise ValueError(
                "Checkpoint LPCR offset mode is "
                f"{saved_offset_activation!r}, but the current run uses "
                f"{cfg.voxelextractor.offset_activation!r}. Pass the matching "
                "--offset-activation value."
            )
        validate_checkpoint_voxel_geometry(saved_voxel_cfg, cfg.voxelextractor)
        ddpm.load_state_dict(checkpoint["weights"])
        ddpm_ema.ema_model.load_state_dict(
            checkpoint.get("ema_weights", checkpoint["weights"])
        )
        ddpm.to(device)
        ddpm_ema.to(device)
        print(
            f"loaded checkpoint {checkpoint_path} "
            f"(stage={checkpoint.get('train_stage')}, epoch={checkpoint.get('epoch')})"
        )
        if is_resume:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1

    if train_stage == 1:
        # The LiDAR encoder/LPCR branch is trained in Stage 1 and frozen while
        # the radar encoder and latent diffusion model are optimized.
        for name, parameter in ddpm.named_parameters():
            if name.startswith("model.preprocess_r."):
                parameter.requires_grad = False

    # =================================================================================
    # Setup optimizer and learning rate scheduler
    # =================================================================================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, ddpm.parameters()),
        lr=learning_rate,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    if is_resume and checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # =================================================================================
    # Training loop
    # =================================================================================


    log_dir = str(logging_dir)
    writer = SummaryWriter(log_dir=log_dir, comment="diffusion")
    global_step = (
        int(checkpoint.get("global_step", 0))
        if is_resume and checkpoint is not None
        else 0
    )
    valid_time = 0
    for epoch in range(start_epoch, max_epochs + 1):

        ddpm.train()
        loss_mean = 0.
        progress_bar = tqdm(
            train_loader,  
            desc="training",
            dynamic_ncols=True,
        )
        for data in progress_bar:
            lidar_path, radar_path, _ = data
            lidar_points = load_data_as_points(str(lidar_path[0]))
            radar_points = load_data_as_points(str(radar_path[0]))
            lidar_example = voxelizer(lidar_points)
            radar_example = voxelizer(radar_points)

            loss = ddpm(
                lidar=lidar_example,
                radar=radar_example,
                lidar_ds=lidar_example,
                grid_size=point_cloud_bounds,
                stage=train_stage,
                global_step=global_step,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            log = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            loss_mean += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{log['lr']:.2e}")
            writer.add_scalars("loss",{"train":loss.item()}, global_step)
            ddpm_ema.update()
            log["ema/decay"] = ddpm_ema.get_current_decay()

        # Record epoch-level training statistics.
        loss_mean = loss_mean / len(train_loader.dataset)
        writer.add_scalars("loss_mean_epoch", {"train": loss_mean}, epoch)
        writer.add_scalars("learning rate", {"train": lr_scheduler.get_last_lr()[0]}, epoch)
        lr_scheduler.step()

        # Validate after every epoch.
        if epoch % 1 == 0:
            loss_val = 0.
            ddpm.eval()
            valid_iter = valid_time * len(valid_loader.dataset)
            progress_bar_valid = tqdm(
                valid_loader,  
                desc="validing",
                dynamic_ncols=True,
            )
            with torch.no_grad():
                for j, data in enumerate(progress_bar_valid):
                    valid_iter = valid_iter + 1
                    lidar_path_v, radar_path_v, _ = data
                    radar_points_v = load_data_as_points(str(radar_path_v[0]))
                    lidar_points_v = load_data_as_points(str(lidar_path_v[0]))
                    radar_example_v = voxelizer(radar_points_v)
                    lidar_example_v = voxelizer(lidar_points_v)

                    valloss = ddpm(
                        lidar=lidar_example_v,
                        radar=radar_example_v,
                        lidar_ds=lidar_example_v,
                        grid_size=point_cloud_bounds,
                        stage=train_stage,
                        global_step=valid_iter + 1,
                    )

                    writer.add_scalars("loss", {"valid": valloss.item()}, valid_iter)
                    loss_val += valloss.item()

                valid_time = valid_time + 1
                val_loss_mean = loss_val / len(valid_loader.dataset)
                writer.add_scalars("loss_mean_epoch", {"valid": val_loss_mean}, epoch)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                    epoch, max_epochs, j + 1, len(valid_loader), val_loss_mean))

        # Save periodic checkpoints and the final model.
        if epoch % cfg.training.save_every_epochs == 0 or epoch == max_epochs:
            save_dir = Path(logging_dir) / "models"
            save_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                {
                    "cfg": dataclasses.asdict(cfg),
                    "weights": ddpm_ema.online_model.state_dict(),
                    "ema_weights": ddpm_ema.ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "global_step": global_step,
                    "stage": stage_name,
                    "train_stage": train_stage,
                    "epoch": epoch,
                },
                save_dir / f"diffusion_{epoch:05d}.pth",
            )
            print(f"diffusion_{epoch:05d}.pth")

    writer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train R2LDM in the LPCR or latent diffusion stage."
    )
    parser.add_argument("--stage", choices=("lpcr", "ldm"), required=True)
    parser.add_argument("--data-root", help="Root of the preprocessed VoD dataset")
    parser.add_argument("--train-split", help="Training split text file")
    parser.add_argument("--val-split", help="Validation split text file")
    parser.add_argument("--output-dir", help="Directory for logs and checkpoints")
    parser.add_argument(
        "--lpcr-checkpoint",
        help="Stage-1 checkpoint used to initialize latent diffusion training",
    )
    parser.add_argument("--resume", help="Checkpoint from the same stage to resume")
    parser.add_argument("--lr", type=float, help="Override the stage learning rate")
    parser.add_argument("--epochs", type=int, help="Override the stage epoch count")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--offset-activation",
        choices=("legacy_sigmoid", "centered_tanh"),
        help=(
            "LPCR offset parameterization. Use legacy_sigmoid for published "
            "checkpoints or centered_tanh for signed center-relative offsets."
        ),
    )
    utils.option.add_voxel_config_arguments(parser)
    return parser


def config_from_args(args: argparse.Namespace) -> utils.option.Config:
    cfg = utils.option.Config()
    if args.data_root is not None:
        cfg.data.root = args.data_root
    if args.train_split is not None:
        cfg.data.train_split = args.train_split
    if args.val_split is not None:
        cfg.data.val_split = args.val_split
    if args.output_dir is not None:
        cfg.training.output_dir = args.output_dir
    if args.lpcr_checkpoint is not None:
        cfg.training.lpcr_checkpoint = args.lpcr_checkpoint
    if args.resume is not None:
        cfg.training.resume_checkpoint = args.resume
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.epochs is not None:
        cfg.training.max_epochs = args.epochs
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
    train(config_from_args(args), stage_name=args.stage)


if __name__ == "__main__":
    main()
