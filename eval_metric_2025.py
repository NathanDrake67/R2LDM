"""Evaluate generated R2LDM point clouds against paired ground truth.

This is a cleaned, path-configurable version of the metric script supplied by
the authors. It reports squared-L2 Chamfer distance, squared-L2 Hausdorff
distance, F-score, and BEV JSD/MMD. PointNet-based Frechet distance is optional
because it downloads a pretrained feature extractor on first use.

Expected layout::

    DATA_ROOT/
    |-- gen/<sequence>/<frame>.bin
    `-- gt/<sequence>/<frame>.bin
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from metrics import bev, distribution
from utils.option import MetricVoxelConfig


def load_point_cloud(path: Path, columns: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % columns:
        raise ValueError(
            f"{path} contains {values.size} float32 values, which is not "
            f"divisible by --*-cols={columns}"
        )
    points = values.reshape(-1, columns)[:, :3]
    return points[np.isfinite(points).all(axis=1)]


def voxel_centroids(
    points: np.ndarray,
    point_cloud_range: np.ndarray,
    voxel_size: np.ndarray,
) -> np.ndarray:
    """Return one XYZ centroid per occupied voxel using NumPy only."""
    lower = point_cloud_range[:3]
    upper = point_cloud_range[3:]
    keep = ((points >= lower) & (points < upper)).all(axis=1)
    points = points[keep]
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)

    indices = np.floor((points - lower) / voxel_size).astype(np.int64)
    _, inverse = np.unique(indices, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    counts = np.zeros(inverse.max() + 1, dtype=np.int64)
    np.add.at(sums, inverse, points)
    np.add.at(counts, inverse, 1)
    return (sums / counts[:, None]).astype(np.float32)


def collect_pairs(
    data_root: Path,
    gen_subdir: str,
    gt_subdir: str,
    sequences: set[str] | None,
) -> tuple[list[tuple[Path, Path, Path]], list[Path]]:
    gen_root = data_root / gen_subdir
    gt_root = data_root / gt_subdir
    if not gen_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError(
            f"Expected generated and ground-truth directories: {gen_root}, {gt_root}"
        )

    pairs: list[tuple[Path, Path, Path]] = []
    missing: list[Path] = []
    for gen_path in sorted(gen_root.rglob("*.bin")):
        relative = gen_path.relative_to(gen_root)
        if sequences and (len(relative.parts) < 2 or relative.parts[0] not in sequences):
            continue
        gt_path = gt_root / relative
        if gt_path.is_file():
            pairs.append((relative, gen_path, gt_path))
        else:
            missing.append(relative)
    return pairs, missing


@torch.inference_mode()
def nearest_squared_distances(
    source: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """For every source point, find squared L2 distance to the nearest target."""
    result: list[torch.Tensor] = []
    for source_start in range(0, source.shape[0], chunk_size):
        source_chunk = source[source_start : source_start + chunk_size]
        nearest = torch.full(
            (source_chunk.shape[0],),
            float("inf"),
            device=source.device,
            dtype=source.dtype,
        )
        for target_start in range(0, target.shape[0], chunk_size):
            target_chunk = target[target_start : target_start + chunk_size]
            distances = torch.cdist(source_chunk, target_chunk, p=2).square()
            nearest = torch.minimum(nearest, distances.amin(dim=1))
        result.append(nearest)
    return torch.cat(result)


@torch.inference_mode()
def geometric_metrics(
    generated: np.ndarray,
    ground_truth: np.ndarray,
    device: torch.device,
    chunk_size: int,
    fscore_threshold: float,
) -> dict[str, float] | None:
    if not len(generated) or not len(ground_truth):
        return None

    generated_tensor = torch.from_numpy(generated).to(device)
    ground_truth_tensor = torch.from_numpy(ground_truth).to(device)
    pred_to_gt = nearest_squared_distances(
        generated_tensor, ground_truth_tensor, chunk_size
    )
    gt_to_pred = nearest_squared_distances(
        ground_truth_tensor, generated_tensor, chunk_size
    )

    threshold_squared = fscore_threshold**2
    precision = (pred_to_gt <= threshold_squared).float().mean()
    recall = (gt_to_pred <= threshold_squared).float().mean()
    fscore = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "chamfer_squared_l2": float((pred_to_gt.mean() + gt_to_pred.mean()).cpu()),
        "hausdorff_squared_l2": float(
            torch.maximum(pred_to_gt.max(), gt_to_pred.max()).cpu()
        ),
        "fscore": float(fscore.cpu()),
        "precision": float(precision.cpu()),
        "recall": float(recall.cpu()),
        "generated_points": int(generated.shape[0]),
        "ground_truth_points": int(ground_truth.shape[0]),
    }


def build_parser() -> argparse.ArgumentParser:
    metric_voxel_config = MetricVoxelConfig()
    parser = argparse.ArgumentParser(
        description="Compute R2LDM point-cloud generation metrics."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--gen-subdir", default="gen")
    parser.add_argument("--gt-subdir", default="gt")
    parser.add_argument(
        "--sequences",
        nargs="*",
        help="Optional sequence names; omission evaluates every matched sequence",
    )
    parser.add_argument("--gen-cols", type=int, default=3)
    parser.add_argument("--gt-cols", type=int, default=3)
    parser.add_argument("--fscore-threshold", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--raw-points",
        action="store_true",
        help="Skip the Jihe voxel-centroid preprocessing used by the supplied script",
    )
    parser.add_argument(
        "--point-cloud-range",
        nargs=6,
        type=float,
        default=metric_voxel_config.point_cloud_range,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
    )
    parser.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        default=metric_voxel_config.voxel_size,
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument("--bev-field-size", type=float, default=160.0)
    parser.add_argument("--bev-bins", type=int, default=100)
    parser.add_argument("--bev-min-depth", type=float, default=3.0)
    parser.add_argument("--bev-max-depth", type=float, default=70.0)
    parser.add_argument(
        "--pointnet-fpd",
        action="store_true",
        help="Also compute PointNet Frechet distance (downloads pretrained weights)",
    )
    parser.add_argument("--pointnet-normalization", type=float, default=80.0)
    parser.add_argument("--compile-pointnet", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metric_results/summary.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gen_cols < 3 or args.gt_cols < 3:
        raise ValueError("--gen-cols and --gt-cols must be at least 3")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.fscore_threshold <= 0:
        raise ValueError("--fscore-threshold must be positive")

    data_root = args.data_root.expanduser().resolve()
    sequences = set(args.sequences) if args.sequences else None
    pairs, missing = collect_pairs(
        data_root, args.gen_subdir, args.gt_subdir, sequences
    )
    if missing:
        warnings.warn(f"Ignoring {len(missing)} generated files without matching GT")
    pairs_discovered = len(pairs)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError("No matching generated/ground-truth .bin pairs were found")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    point_cloud_range = np.asarray(args.point_cloud_range, dtype=np.float32)
    voxel_size = np.asarray(args.voxel_size, dtype=np.float32)

    pointnet = None
    real_features: list[np.ndarray] = []
    generated_features: list[np.ndarray] = []
    if args.pointnet_fpd:
        from metrics.extractor.pointnet import pretrained_pointnet

        pointnet = pretrained_pointnet(
            dataset="shapenet",
            device=device,
            compile=args.compile_pointnet,
        )

    per_frame: list[dict[str, float | int | str]] = []
    real_histograms: list[torch.Tensor] = []
    generated_histograms: list[torch.Tensor] = []
    skipped_empty = 0

    for relative, gen_path, gt_path in tqdm(pairs, desc="evaluating"):
        generated = load_point_cloud(gen_path, args.gen_cols)
        ground_truth = load_point_cloud(gt_path, args.gt_cols)
        if not args.raw_points:
            generated = voxel_centroids(generated, point_cloud_range, voxel_size)
            ground_truth = voxel_centroids(ground_truth, point_cloud_range, voxel_size)

        values = geometric_metrics(
            generated,
            ground_truth,
            device=device,
            chunk_size=args.chunk_size,
            fscore_threshold=args.fscore_threshold,
        )
        if values is None:
            skipped_empty += 1
            continue
        values["sample"] = relative.as_posix()
        per_frame.append(values)

        gt_hist = bev.point_cloud_to_histogram(
            torch.from_numpy(ground_truth),
            field_size=args.bev_field_size,
            bins=args.bev_bins,
            min_depth=args.bev_min_depth,
            max_depth=args.bev_max_depth,
        )
        gen_hist = bev.point_cloud_to_histogram(
            torch.from_numpy(generated),
            field_size=args.bev_field_size,
            bins=args.bev_bins,
            min_depth=args.bev_min_depth,
            max_depth=args.bev_max_depth,
        )
        if gt_hist.sum() > 0 and gen_hist.sum() > 0:
            real_histograms.append(gt_hist)
            generated_histograms.append(gen_hist)

        if pointnet is not None:
            gt_tensor = torch.from_numpy(ground_truth).T.unsqueeze(0).to(device)
            gen_tensor = torch.from_numpy(generated).T.unsqueeze(0).to(device)
            with torch.inference_mode():
                real_features.append(
                    pointnet(gt_tensor / args.pointnet_normalization).cpu().numpy()
                )
                generated_features.append(
                    pointnet(gen_tensor / args.pointnet_normalization).cpu().numpy()
                )

    if not per_frame:
        raise RuntimeError("Every matched pair was empty after preprocessing")

    averaged_keys = (
        "chamfer_squared_l2",
        "hausdorff_squared_l2",
        "fscore",
        "precision",
        "recall",
        "generated_points",
        "ground_truth_points",
    )
    summary: dict[str, object] = {
        "pairs_discovered": pairs_discovered,
        "pairs_evaluated": len(per_frame),
        "pairs_skipped_empty": skipped_empty,
        "distance_convention": "squared L2",
        "fscore_threshold": args.fscore_threshold,
        "voxelization": None
        if args.raw_points
        else {
            "point_cloud_range": list(args.point_cloud_range),
            "voxel_size": list(args.voxel_size),
        },
        "mean": {
            key: float(np.mean([float(row[key]) for row in per_frame]))
            for key in averaged_keys
        },
    }

    if real_histograms:
        real_bev = torch.stack(real_histograms).to(device).float()
        generated_bev = torch.stack(generated_histograms).to(device).float()
        summary["bev"] = {
            "valid_pairs": len(real_histograms),
            "jsd": bev.compute_jsd_2d(real_bev, generated_bev),
            "mmd": bev.compute_mmd_2d(real_bev, generated_bev),
        }

    if pointnet is not None:
        if len(real_features) < 2:
            warnings.warn("PointNet Frechet distance needs at least two valid pairs")
        else:
            summary["pointnet_frechet_distance"] = (
                distribution.compute_frechet_distance(
                    np.concatenate(real_features),
                    np.concatenate(generated_features),
                )
            )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    per_frame_path = output_path.with_name(f"{output_path.stem}.per_frame.csv")
    with per_frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Per-frame metrics: {per_frame_path}")


if __name__ == "__main__":
    main()
