"""Train the paper VHA+ assessor from scored High/Low Q/A manifests."""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional
from scipy.stats import pearsonr, spearmanr
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataloader import (
    TRAINING_ROOT,
    BENCHMARK_ROOT,
    RuntimeTargetMixer,
    TargetStatistics,
    TrainingSubset,
    build_training_data,
    compute_target_statistics,
    validate_training_inputs,
)

DINO_CHECKPOINT_FORMAT = "dinov3-cls-trainable-v1"


class DinoV3ClsMetric(nn.Module):
    """Final normalized DINOv3 CLS cosine distance with an affine head."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.visual = timm.create_model(args.dinov3_model, pretrained=True)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (args.input_size, args.input_size),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        if len(self.visual.blocks) != 12 or not 0 <= args.trainable_block_start < 12:
            raise ValueError(
                "Expected DINOv3 ViT-B with 12 blocks and a valid trainable start"
            )
        self.visual.requires_grad_(False)
        for block in self.visual.blocks[args.trainable_block_start :]:
            block.requires_grad_(True)
        self.visual.set_grad_checkpointing(args.gradient_checkpointing)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.shift = nn.Parameter(torch.zeros(()))
        self.block_start = args.trainable_block_start
        self.cls_features = "final"

    def encode_cls(self, pixels):
        tokens = self.visual.forward_features(pixels)
        return functional.normalize(tokens[:, 0].float(), dim=-1)

    def compare(self, gt, candidate):
        distance = 1.0 - (gt * candidate).sum(dim=-1)
        return (self.log_scale.exp() * distance + self.shift, distance)

    def forward(self, gt, candidate):
        features = self.encode_cls(torch.cat([gt, candidate])).chunk(2)
        return self.compare(features[0], features[1])

    def trainable_summary(self) -> dict[str, Any]:
        return {
            "parameters_total": sum((p.numel() for p in self.parameters())),
            "parameters_trainable": sum(
                (p.numel() for p in self.parameters() if p.requires_grad)
            ),
            "text_tower_parameters": 0,
            "trainable_block_start": self.block_start,
            "final_head": "distance",
            "comparison_head_parameters": 0,
            "cls_features": "final",
            "cls_block_indices": [11],
            "cls_feature_dim": self.visual.num_features * 1,
            "cls_distance_weight": 1.0,
            "spatial_distance_weight": 0.0,
        }


def setup_distributed() -> tuple[int, int, int, Any]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return (rank, world, local_rank, torch.device("cuda", local_rank))


def safe_corr(function: Any, x: Any, y: Any) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    result = function(x, y)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def metric_summary(rows: list[dict[str, Any]], human: bool = False) -> dict[str, float]:
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
    raw_distance = np.asarray([row["raw_distance"] for row in rows], dtype=np.float64)
    result = {
        "pairs": float(len(rows)),
        "pooled_spearman": safe_corr(spearmanr, target, prediction),
        "pooled_pearson": safe_corr(pearsonr, target, prediction),
        "raw_distance_pooled_spearman": safe_corr(spearmanr, target, raw_distance),
        "raw_distance_pooled_pearson": safe_corr(pearsonr, target, raw_distance),
        "mse": float(np.mean((prediction - target) ** 2)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }
    result["raw_distance_pooled_spearman"] = result["raw_distance_pooled_spearman"]
    result["raw_distance_pooled_pearson"] = result["raw_distance_pooled_pearson"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(row)
    correlations = []
    correct = total = 0
    for image_rows in grouped.values():
        y = np.asarray([row["target"] for row in image_rows])
        p = np.asarray([row["prediction"] for row in image_rows])
        value = safe_corr(spearmanr, y, p)
        if math.isfinite(value):
            correlations.append(value)
        for left in range(len(image_rows)):
            for right in range(left + 1, len(image_rows)):
                target_delta = y[left] - y[right]
                prediction_delta = p[left] - p[right]
                if target_delta == 0:
                    continue
                total += 1
                correct += int(target_delta * prediction_delta > 0)
    result["image_mean_spearman"] = float(np.mean(correlations))
    result["pairwise_accuracy"] = correct / total if total else float("nan")
    result["image_spearman_count"] = float(len(correlations))
    result["pairwise_comparisons"] = float(total)
    if not human:
        for family in ("lr", "pisasr", "pisasr_2x"):
            subset = [row for row in rows if row.get("family") == family]
            result[f"{family}_pooled_spearman"] = safe_corr(
                spearmanr,
                np.asarray([row["target"] for row in subset]),
                np.asarray([row["prediction"] for row in subset]),
            )
    return result


def gather_rows(
    local_rows: list[dict[str, Any]], rank: int, world: int
) -> list[dict[str, Any]]:
    if world == 1:
        return local_rows
    gathered: list[Any] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_rows)
    if rank != 0:
        return []
    rows = [row for shard in gathered for row in shard]
    unique = {row["sample_id"]: row for row in rows}
    if len(unique) != len(rows):
        raise RuntimeError(f"Duplicate evaluation rows: {len(rows) - len(unique)}")
    return list(unique.values())


def evaluate(
    model: Any,
    loader: Any,
    device: Any,
    amp_dtype: Any,
    target_mean: float,
    target_std: float,
    rank: int,
    world: int,
    human: bool = False,
) -> tuple[dict[str, float] | None, list[dict[str, Any]]]:
    model.eval()
    local_rows = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda", dtype=amp_dtype):
                prediction_z, raw_distance = model(
                    batch["gt"].to(device, non_blocking=True),
                    batch["candidate"].to(device, non_blocking=True),
                )
            prediction = prediction_z.float().cpu().numpy() * target_std + target_mean
            for row, pred, raw in zip(
                batch["rows"], prediction, raw_distance.float().cpu().numpy()
            ):
                local_rows.append(
                    {
                        **row,
                        "target": float(
                            row["human_score"] if human else row["vqa_target_raw"]
                        ),
                        "prediction": float(pred),
                        "raw_distance": float(raw),
                    }
                )
    rows = gather_rows(local_rows, rank, world)
    return (metric_summary(rows, human=human) if rank == 0 else None, rows)


def batch_pearson(prediction: Any, target: Any) -> Any:
    centered_p = prediction - prediction.mean()
    centered_t = target - target.mean()
    return (centered_p * centered_t).mean() / (
        centered_p.square().mean().sqrt() * centered_t.square().mean().sqrt() + 1e-08
    )


def regression_loss(
    prediction: Any, target: Any, corr_weight: float
) -> tuple[Any, dict[str, float]]:
    mse = (prediction - target).square().mean()
    correlation = batch_pearson(prediction, target)
    loss = mse + corr_weight * (1.0 - correlation)
    return (loss, {"mse": float(mse.detach()), "pearson": float(correlation.detach())})


def raw_vqa_reporting_scale(
    statistics: TargetStatistics, args: argparse.Namespace
) -> tuple[float, float]:
    """Map the model's normalized output back to the configured raw VQA target."""
    if not math.isclose(
        args.direct_target_weight, 0.0, abs_tol=1e-12
    ) or not math.isclose(args.vqa_target_weight, 1.0, abs_tol=1e-12):
        raise RuntimeError("Checkpoint targets must use VHA scores only.")
    output_mean = statistics.mixed_mean * statistics.vqa_std + statistics.vqa_mean
    output_std = statistics.mixed_std * statistics.vqa_std
    return (output_mean, output_std)


def unwrap_model(model: Any) -> Any:
    return model.module if isinstance(model, DistributedDataParallel) else model


def trainable_parameter_names(model: Any) -> tuple[str, ...]:
    module = unwrap_model(model)
    return tuple(
        sorted(
            (
                name
                for name, parameter in module.named_parameters()
                if parameter.requires_grad
            )
        )
    )


def checkpoint_model_config(args):
    return {
        "backbone": "dinov3",
        "dinov3_model": args.dinov3_model,
        "input_size": args.input_size,
        "trainable_block_start": args.trainable_block_start,
        "final_head": "distance",
        "timm_version": timm.__version__,
        "features": "final-post-norm-cls-l2-normalized",
        "cls_features": "final",
        "cls_block_indices": [11],
    }


def save_checkpoint(
    model: Any,
    optimizer: Any,
    step: int,
    args: argparse.Namespace,
    path: Path,
    *,
    kind: str,
    include_optimizer: bool,
    best_metric: float,
    best_step: int,
    epoch: int,
    batch_in_epoch: int,
    world_size: int,
) -> None:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    names = trainable_parameter_names(module)
    full_state = module.state_dict()
    missing = [name for name in names if name not in full_state]
    if missing:
        raise RuntimeError(f"Trainable parameters missing from model state: {missing}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "checkpoint_format": DINO_CHECKPOINT_FORMAT,
        "checkpoint_kind": kind,
        "step": step,
        "trainable_model": {
            name: full_state[name].detach().cpu().clone() for name in names
        },
        "trainable_parameter_names": list(names),
        "model_config": checkpoint_model_config(args),
        "training_state": {
            "best_validation_image_mean_spearman": best_metric,
            "best_global_step": best_step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "world_size": world_size,
        },
        "args": vars(args),
        "target_statistics": getattr(args, "target_statistics", None),
        "reporting_scale": getattr(args, "reporting_scale", None),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, temporary)
    os.replace(temporary, path)
    path.chmod(438)


def load_checkpoint(
    model: Any,
    optimizer: Any | None,
    path: Path,
    args: argparse.Namespace,
    *,
    require_optimizer: bool,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    module = unwrap_model(model)
    stored_args = checkpoint.get("args", {})
    for key in (
        "reverse_weight",
        "reverse_gate_threshold",
        "reverse_margin_threshold",
        "vqa_target_weight",
        "direct_target_weight",
        "data_root",
        "forward_support_threshold",
    ):
        if str(stored_args.get(key)) != str(getattr(args, key, None)):
            raise RuntimeError(f"Checkpoint target configuration mismatch: {key}")
    subset_mismatches = {}
    for key in ("candidate_sets", "downscale_sets"):
        if key not in stored_args:
            continue
        stored = tuple(stored_args[key])
        current = tuple(getattr(args, key))
        if stored != current:
            subset_mismatches[key] = {"checkpoint": stored, "current": current}
    if subset_mismatches:
        raise RuntimeError(f"Checkpoint training-subset mismatch: {subset_mismatches}")
    stored_ranking_type = stored_args.get("ranking_loss_type", "pearson")
    if stored_ranking_type != "pearson":
        raise RuntimeError(
            f"Checkpoint ranking-loss mismatch: checkpoint={stored_ranking_type}, current={'pearson'}"
        )
    stored_ranking_weight = stored_args.get(
        "ranking_loss_weight", stored_args.get("correlation_weight")
    )
    if stored_ranking_weight is not None and (
        not math.isclose(
            float(stored_ranking_weight), args.ranking_loss_weight, abs_tol=1e-12
        )
    ):
        raise RuntimeError(
            f"Checkpoint ranking-loss weight mismatch: checkpoint={stored_ranking_weight}, current={args.ranking_loss_weight}"
        )
    expected_format = DINO_CHECKPOINT_FORMAT
    if checkpoint.get("checkpoint_format") == expected_format:
        expected_config = checkpoint_model_config(args)
        stored_config = dict(checkpoint.get("model_config", {}))
        stored_config.setdefault("final_head", "distance")
        stored_config.setdefault("cls_features", "final")
        stored_config.setdefault("cls_block_indices", [11])
        mismatched = {
            key: {"checkpoint": stored_config.get(key), "current": value}
            for key, value in expected_config.items()
            if stored_config.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                f"Checkpoint base-model configuration mismatch: {mismatched}"
            )
        expected_names = set(trainable_parameter_names(module))
        stored_state = checkpoint.get("trainable_model", {})
        stored_names = set(stored_state)
        if stored_names != expected_names:
            raise RuntimeError(
                f"Checkpoint trainable-parameter mismatch: missing={sorted(expected_names - stored_names)}, unexpected={sorted(stored_names - expected_names)}"
            )
        current_state = module.state_dict()
        with torch.no_grad():
            for name in sorted(expected_names):
                source = stored_state[name]
                target = current_state[name]
                if source.shape != target.shape:
                    raise RuntimeError(
                        f"Checkpoint shape mismatch for {name}: checkpoint={tuple(source.shape)}, current={tuple(target.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))
    elif "model" in checkpoint:
        module.load_state_dict(checkpoint["model"], strict=True)
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    if require_optimizer:
        if optimizer is None or "optimizer" not in checkpoint:
            raise RuntimeError(
                f"Checkpoint cannot resume training because optimizer state is absent: {path}"
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        parameter_device = next(module.parameters()).device
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(parameter_device)
    return {
        "step": int(checkpoint.get("step", 0)),
        **checkpoint.get("training_state", {}),
    }


def train_main(args: argparse.Namespace) -> None:
    os.umask(0)
    if args.save_every_iterations <= 0:
        raise ValueError("--save-every-iterations must be positive")
    for option, value in (
        ("--backbone-learning-rate", args.backbone_learning_rate),
        ("--scale-learning-rate", args.scale_learning_rate),
    ):
        if value < 0:
            raise ValueError(f"{option} must be non-negative")
    if args.ranking_loss_weight < 0:
        raise ValueError("--ranking-loss-weight must be non-negative")
    if args.resume is not None and (not args.resume.is_file()):
        raise FileNotFoundError(args.resume)
    forward_thresholding = args.forward_support_threshold is not None
    if forward_thresholding and math.isnan(args.forward_support_threshold):
        raise ValueError("--forward-support-threshold must not be NaN")
    raw_summary = validate_training_inputs(
        args.data_root,
        args.human_test_manifest,
        require_forward_scores=forward_thresholding,
    )
    training_subset = TrainingSubset(
        candidate_sets=args.candidate_sets, downscale_sets=args.downscale_sets
    )
    rank, world, local_rank, device = setup_distributed()
    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    reproducibility = {
        "base_seed": args.seed,
        "process_seed": process_seed,
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
        "cuda_seeded": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "dataloader_worker_seeded": True,
        "deterministic_algorithms": False,
    }
    statistics_payload: list[Any] = [None]
    if rank == 0:
        statistics_payload[0] = compute_target_statistics(
            args.data_root / "train_raw.csv",
            training_subset=training_subset,
            reverse_gate_threshold=args.reverse_gate_threshold,
            reverse_margin_threshold=args.reverse_margin_threshold,
            reverse_weight=args.reverse_weight,
            vqa_target_weight=args.vqa_target_weight,
            direct_target_weight=args.direct_target_weight,
            forward_support_threshold=args.forward_support_threshold,
        ).as_dict()
    if world > 1:
        dist.broadcast_object_list(statistics_payload, src=0)
    statistics = TargetStatistics(**statistics_payload[0])
    args.target_statistics = statistics.as_dict()
    target_mixer = RuntimeTargetMixer(
        reverse_gate_threshold=args.reverse_gate_threshold,
        reverse_margin_threshold=args.reverse_margin_threshold,
        reverse_weight=args.reverse_weight,
        vqa_target_weight=args.vqa_target_weight,
        direct_target_weight=args.direct_target_weight,
        statistics=statistics,
        forward_support_threshold=args.forward_support_threshold,
    )
    target_definition = target_mixer.summary()
    target_definition["raw_score_divisor"] = raw_summary.get("vqa_divisor", 1)
    report_output_mean, report_output_std = raw_vqa_reporting_scale(statistics, args)
    args.reporting_scale = {"mean": report_output_mean, "std": report_output_std}
    model = DinoV3ClsMetric(args)
    preprocess = model.preprocess
    model = model.to(device)
    data = build_training_data(
        data_root=args.data_root,
        human_test_manifest=args.human_test_manifest,
        training_subset=training_subset,
        target_mixer=target_mixer,
        preprocess=preprocess,
        batch_size=args.batch_size_per_gpu,
        workers=args.workers,
        seed=args.seed,
        rank=rank,
        world=world,
    )
    train_loader = data.train_loader
    val_loader = data.validation_loader
    human_loader = data.human_test_loader
    train_sampler = data.train_sampler
    training_subset_summary = data.training_subset_summary
    if rank == 0:
        print(
            "TRAINING_SUBSET",
            json.dumps(training_subset_summary, sort_keys=True),
            flush=True,
        )
    trainable = model.trainable_summary()
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("visual.")
    ]
    scale_parameters = [model.log_scale, model.shift]
    optimizer_groups = [
        {
            "params": backbone_parameters,
            "lr": args.backbone_learning_rate,
            "weight_decay": args.weight_decay,
            "group_name": "dinov3",
        }
    ]
    optimizer_groups.append(
        {
            "params": scale_parameters,
            "lr": args.scale_learning_rate,
            "weight_decay": 0.0,
            "group_name": "affine",
        }
    )
    optimizer = torch.optim.AdamW(optimizer_groups)
    resume_state: dict[str, Any] = {}
    if args.resume is not None:
        resume_state = load_checkpoint(
            model, optimizer, args.resume, args, require_optimizer=True
        )
        saved_world_size = int(resume_state.get("world_size", world))
        if saved_world_size != world:
            raise RuntimeError(
                f"Training resume requires the original world size: checkpoint={saved_world_size}, current={world}"
            )
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        config = {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "train_samples": data.train_samples,
            "validation_samples": data.validation_samples,
            "realsr_samples": data.human_test_samples,
            "target_definition": target_definition,
            "training_subset": training_subset_summary,
            "reproducibility": reproducibility,
            "output_score": {
                "space": "raw_vqa_target",
                "mean": report_output_mean,
                "std": report_output_std,
                "statistics_source": "train_raw.csv",
            },
            **trainable,
        }
        (args.output_dir / "run_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
    history = args.output_dir / "metrics.jsonl"
    baseline_validation, baseline_rows = evaluate(
        model,
        val_loader,
        device,
        amp_dtype,
        report_output_mean,
        report_output_std,
        rank,
        world,
    )
    baseline_test, _ = evaluate(
        model,
        human_loader,
        device,
        amp_dtype,
        report_output_mean,
        report_output_std,
        rank,
        world,
        human=True,
    )
    initial_step = int(resume_state.get("step", 0))
    initial_epoch = int(resume_state.get("epoch", initial_step // len(train_loader)))
    initial_batch_in_epoch = int(
        resume_state.get("batch_in_epoch", initial_step % len(train_loader))
    )
    resumed_best = float(
        resume_state.get("best_validation_image_mean_spearman", -float("inf"))
    )
    resumed_best_step = int(resume_state.get("best_global_step", initial_step))
    if rank == 0:
        baseline_record = {
            "step": initial_step,
            "split": "resume_baseline"
            if args.resume is not None
            else f"frozen_{'dinov3'}_cls_baseline",
            "validation": baseline_validation,
            "realsr_test": baseline_test,
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(baseline_record) + "\n")
        if args.resume is None:
            save_checkpoint(
                model,
                optimizer,
                0,
                args,
                args.output_dir / "best.pt",
                kind="best",
                include_optimizer=False,
                best_metric=baseline_validation["image_mean_spearman"],
                best_step=0,
                epoch=0,
                batch_in_epoch=0,
                world_size=world,
            )
        print("BASELINE", json.dumps(baseline_record, sort_keys=True), flush=True)
    best = (
        max(resumed_best, baseline_validation["image_mean_spearman"])
        if rank == 0
        else -float("inf")
    )
    best_step = (
        resumed_best_step
        if rank == 0 and resumed_best >= baseline_validation["image_mean_spearman"]
        else initial_step
    )
    step = initial_step
    epoch = initial_epoch
    resume_epoch = initial_epoch
    resume_batch_in_epoch = initial_batch_in_epoch
    start = time.time()
    while step < args.max_training_iterations:
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if epoch == resume_epoch and batch_index < resume_batch_in_epoch:
                continue
            if step >= args.max_training_iterations:
                break
            step += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                gt = batch["gt"].to(device, non_blocking=True)
                prediction, raw_distance = model(
                    gt, batch["candidate"].to(device, non_blocking=True)
                )
                target = batch["target"].to(device, non_blocking=True)
                loss_function = regression_loss
                loss, components = loss_function(
                    prediction.float(), target, args.ranking_loss_weight
                )
                training_rows = batch["rows"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            if step % args.log_every_steps == 0 and rank == 0:
                module = (
                    model.module
                    if isinstance(model, DistributedDataParallel)
                    else model
                )
                raw_predictions = (
                    prediction.detach().float().cpu().numpy() * report_output_std
                    + report_output_mean
                )
                raw_targets = np.asarray(
                    [float(row["vqa_target_raw"]) for row in training_rows],
                    dtype=np.float64,
                )
                record = {
                    "train/loss": float(loss.detach()),
                    "train/mse": components["mse"],
                    "train/raw_target_mean": float(raw_targets.mean()),
                    "train/raw_prediction_mean": float(raw_predictions.mean()),
                    "train/raw_mse": float(
                        np.mean((raw_predictions - raw_targets) ** 2)
                    ),
                    "train/grad_norm": float(grad_norm),
                    "train/raw_distance_mean": float(raw_distance.detach().mean()),
                    "train/affine_scale": float(module.log_scale.detach().exp()),
                    "train/affine_shift": float(module.shift.detach()),
                    "train/steps_per_second": step / (time.time() - start),
                }
                record["train/pearson"] = components["pearson"]
                record["train/ranking_loss"] = 1.0 - components["pearson"]
                with history.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "training": record}) + "\n")
                print("TRAIN", step, json.dumps(record, sort_keys=True), flush=True)
            validate = (
                step % args.validation_every_iterations == 0
                or step == args.max_training_iterations
            )
            test_realsr = (
                step % args.test_every_iterations == 0
                or step == args.max_training_iterations
            )
            if validate or test_realsr:
                validation = validation_rows = None
                test = None
                if validate:
                    validation, validation_rows = evaluate(
                        model,
                        val_loader,
                        device,
                        amp_dtype,
                        report_output_mean,
                        report_output_std,
                        rank,
                        world,
                    )
                if test_realsr:
                    test, _ = evaluate(
                        model,
                        human_loader,
                        device,
                        amp_dtype,
                        report_output_mean,
                        report_output_std,
                        rank,
                        world,
                        human=True,
                    )
                if rank == 0:
                    record = {"step": step}
                    if validation is not None:
                        record["validation"] = validation
                    if test is not None:
                        record["realsr_test"] = test
                    with history.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    if validation is not None:
                        if validation["image_mean_spearman"] > best:
                            best = validation["image_mean_spearman"]
                            best_step = step
                            save_checkpoint(
                                model,
                                optimizer,
                                step,
                                args,
                                args.output_dir / "best.pt",
                                kind="best",
                                include_optimizer=False,
                                best_metric=best,
                                best_step=best_step,
                                epoch=epoch,
                                batch_in_epoch=batch_index + 1,
                                world_size=world,
                            )
                    print("EVALUATION", json.dumps(record, sort_keys=True), flush=True)
                model.train()
            save_periodic = (
                step % args.save_every_iterations == 0
                or step == args.max_training_iterations
            )
            if save_periodic and rank == 0:
                next_epoch = epoch
                next_batch_in_epoch = batch_index + 1
                if next_batch_in_epoch >= len(train_loader):
                    next_epoch += 1
                    next_batch_in_epoch = 0
                periodic_path = checkpoint_dir / f"step_{step:08d}.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    args,
                    periodic_path,
                    kind="periodic_weights",
                    include_optimizer=False,
                    best_metric=best,
                    best_step=best_step,
                    epoch=next_epoch,
                    batch_in_epoch=next_batch_in_epoch,
                    world_size=world,
                )
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    args,
                    args.output_dir / "last.pt",
                    kind="resume",
                    include_optimizer=True,
                    best_metric=best,
                    best_step=best_step,
                    epoch=next_epoch,
                    batch_in_epoch=next_batch_in_epoch,
                    world_size=world,
                )
                print(
                    f"CHECKPOINT step={step} weights={periodic_path} resume={args.output_dir / 'last.pt'}",
                    flush=True,
                )
        epoch += 1
        resume_batch_in_epoch = 0
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the DINOv3-based VHA+ assessor."
    )
    parser.add_argument(
        "--raw-score-manifest-root",
        dest="data_root",
        type=Path,
        default=TRAINING_ROOT / "manifests",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vhaplus"))
    parser.add_argument("--resume", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--benchmark", dest="human_test_manifest", type=Path, default=BENCHMARK_ROOT
    )
    source.add_argument("--human-test-manifest", dest="human_test_manifest", type=Path)
    parser.add_argument(
        "--dinov3-model",
        default="hf-hub:timm/vit_base_patch16_dinov3.lvd1689m@c6a5fb7d12bbd3cf3b0079253141c3332aaed7da",
    )
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--trainable-block-start", type=int, default=8)
    parser.add_argument("--max-training-iterations", type=int, default=20000)
    parser.add_argument("--save-every-iterations", type=int, default=1000)
    parser.add_argument("--validation-every-iterations", type=int, default=1000)
    parser.add_argument("--test-every-iterations", type=int, default=100)
    parser.add_argument("--batch-size-per-gpu", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-06)
    parser.add_argument("--scale-learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--correlation-weight", dest="ranking_loss_weight", type=float, default=3.0
    )
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.set_defaults(
        backbone="dinov3",
        cls_features="final",
        final_head="distance",
        ranking_loss_type="pearson",
        gradient_checkpointing=True,
        forward_support_threshold=-2.0,
        reverse_gate_threshold=0.0,
        reverse_margin_threshold=0.5,
        reverse_weight=1.0,
        vqa_target_weight=1.0,
        direct_target_weight=0.0,
        candidate_sets=("pisasr", "pisasr2x"),
        downscale_sets=("0.5", "0.375", "0.25"),
    )
    return parser.parse_args()


def main():
    train_main(parse_args())


if __name__ == "__main__":
    main()
