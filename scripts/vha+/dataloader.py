"""Scored training manifests, target calculation, and image preprocessing."""

from __future__ import annotations
import csv
import json
import math
import random
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.vha.dataloader import BenchmarkDataset, read_pairs
from scripts.vha.dataloader import BENCHMARK_ROOT as BENCHMARK_ROOT

TRAINING_ROOT = Path(__file__).resolve().parents[2] / "data/training"
import numpy as np
from PIL import Image, PngImagePlugin

PNG_TEXT_CHUNK_LIMIT = 64 * 1024 * 1024
PNG_TEXT_MEMORY_LIMIT = 256 * 1024 * 1024
CANDIDATE_SET_TO_FAMILY = {"pisasr": "pisasr", "pisasr2x": "pisasr_2x"}
DOWNSCALE_TO_TAG = {"0.5": "0p5", "0.375": "0p375", "0.25": "0p25"}
CANDIDATE_VARIANTS_PER_SCALE = {"pisasr": 3, "pisasr2x": 3}


def configure_png_metadata_limits() -> None:
    """Allow trusted dataset PNGs with large embedded ICC profiles."""
    PngImagePlugin.MAX_TEXT_CHUNK = max(
        PngImagePlugin.MAX_TEXT_CHUNK, PNG_TEXT_CHUNK_LIMIT
    )
    PngImagePlugin.MAX_TEXT_MEMORY = max(
        PngImagePlugin.MAX_TEXT_MEMORY, PNG_TEXT_MEMORY_LIMIT
    )


configure_png_metadata_limits()


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / self.count)


@dataclass(frozen=True)
class TargetStatistics:
    vqa_mean: float
    vqa_std: float
    direct_mean: float | None
    direct_std: float | None
    mixed_mean: float
    mixed_std: float

    def as_dict(self) -> dict[str, float | None]:
        return {
            "vqa_mean": self.vqa_mean,
            "vqa_std": self.vqa_std,
            "direct_mean": self.direct_mean,
            "direct_std": self.direct_std,
            "mixed_mean": self.mixed_mean,
            "mixed_std": self.mixed_std,
        }


@dataclass(frozen=True)
class TrainingSubset:
    candidate_sets: tuple[str, ...]
    downscale_sets: tuple[str, ...]
    families: frozenset[str] = field(init=False)
    scale_tags: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "families",
            frozenset(
                (CANDIDATE_SET_TO_FAMILY[value] for value in self.candidate_sets)
            ),
        )
        object.__setattr__(
            self,
            "scale_tags",
            frozenset((DOWNSCALE_TO_TAG[value] for value in self.downscale_sets)),
        )

    @property
    def expected_pairs_per_gt(self) -> int:
        variants = sum(
            (CANDIDATE_VARIANTS_PER_SCALE[value] for value in self.candidate_sets)
        )
        return variants * len(self.downscale_sets)

    def matches(self, row: dict[str, Any]) -> bool:
        return row["family"] in self.families and row["scale_tag"] in self.scale_tags

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_sets": list(self.candidate_sets),
            "manifest_families": sorted(self.families),
            "downscale_sets": list(self.downscale_sets),
            "scale_tags": sorted(self.scale_tags),
            "expected_pairs_per_gt": self.expected_pairs_per_gt,
        }


def reverse_penalty_from_raw_scores(
    row: dict[str, Any], reverse_gate_threshold: float, reverse_margin_threshold: float
) -> float:
    if not int(row["reverse_available"]):
        return 0.0
    try:
        candidate_scores = [
            float(value) for value in json.loads(row["reverse_candidate_scores"])
        ]
        gt_scores = [float(value) for value in json.loads(row["reverse_gt_scores"])]
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid reverse Q/A raw scores: {row.get('sample_id', '<unknown>')}"
        ) from error
    n_qa = int(row["reverse_n_qa"])
    if n_qa <= 0 or len(candidate_scores) != n_qa or len(gt_scores) != n_qa:
        raise RuntimeError(
            f"Reverse Q/A raw-score count mismatch for {row.get('sample_id', '<unknown>')}: n_qa={n_qa}, candidate={len(candidate_scores)}, gt={len(gt_scores)}"
        )

    def contribution(candidate_score: float, gt_score: float) -> float:
        if candidate_score <= reverse_gate_threshold:
            return 0.0
        margin = candidate_score - gt_score
        if reverse_margin_threshold >= 0:
            return max(margin - reverse_margin_threshold, 0.0)
        return max(margin, reverse_margin_threshold)

    contributions = (
        contribution(candidate_score, gt_score)
        for candidate_score, gt_score in zip(candidate_scores, gt_scores)
    )
    return sum(contributions) / n_qa


def forward_loss_from_raw_scores(
    row: dict[str, Any], forward_support_threshold: float | None = None
) -> float:
    if forward_support_threshold is None:
        return float(row["forward_loss"])
    if math.isnan(forward_support_threshold):
        raise RuntimeError("Forward support threshold must not be NaN")
    try:
        gt_scores = [float(value) for value in json.loads(row["forward_gt_scores"])]
        candidate_scores = [
            float(value) for value in json.loads(row["forward_candidate_scores"])
        ]
        n_qa = int(row["forward_n_qa"])
    except (KeyError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Missing or invalid per-Q/A forward scores: {row.get('sample_id', '<unknown>')}. Use a new scored-Q/A manifest directory."
        ) from error
    if n_qa < 0 or len(gt_scores) != n_qa or len(candidate_scores) != n_qa:
        raise RuntimeError(
            f"Forward Q/A raw-score count mismatch: {row.get('sample_id', '<unknown>')}"
        )
    if not all((math.isfinite(v) for v in gt_scores + candidate_scores)):
        raise RuntimeError(
            f"Nonfinite forward Q/A score: {row.get('sample_id', '<unknown>')}"
        )
    return (
        sum(
            (
                gt - candidate
                for gt, candidate in zip(gt_scores, candidate_scores)
                if gt > forward_support_threshold
            )
        )
        / n_qa
        if n_qa
        else 0.0
    )


def raw_vqa_target(
    row: dict[str, Any],
    reverse_gate_threshold: float,
    reverse_margin_threshold: float,
    reverse_weight: float,
    forward_support_threshold: float | None = None,
) -> tuple[float, float]:
    forward_loss = forward_loss_from_raw_scores(row, forward_support_threshold)
    reverse_penalty = reverse_penalty_from_raw_scores(
        row,
        reverse_gate_threshold=reverse_gate_threshold,
        reverse_margin_threshold=reverse_margin_threshold,
    )
    divisor = float(row.get("vqa_divisor", 1.0))
    if not math.isfinite(divisor) or divisor <= 0:
        raise RuntimeError("VQA target divisor must be finite and positive")
    return (
        (forward_loss + reverse_weight * reverse_penalty) / divisor,
        reverse_penalty,
    )


def validate_target_options(
    reverse_gate_threshold: float,
    reverse_margin_threshold: float,
    vqa_target_weight: float,
    direct_target_weight: float,
) -> None:
    if math.isnan(reverse_gate_threshold):
        raise RuntimeError("Reverse gate threshold must not be NaN")
    if math.isnan(reverse_margin_threshold) or reverse_margin_threshold == float("inf"):
        raise RuntimeError(
            "Reverse margin threshold must not be NaN or positive infinity"
        )
    if vqa_target_weight < 0 or direct_target_weight < 0:
        raise RuntimeError("Target fusion weights must be non-negative")
    if not math.isclose(vqa_target_weight + direct_target_weight, 1.0, abs_tol=1e-09):
        raise RuntimeError("VQA and direct target weights must sum to one")


def validate_training_inputs(
    data_root: Path, human_test_manifest: Path, *, require_forward_scores: bool = False
) -> dict[str, Any]:
    """Fail before distributed setup when a canonical training input is absent."""
    required = {
        "train raw-score manifest": data_root / "train_raw.csv",
        "validation raw-score manifest": data_root / "val_raw.csv",
        "raw-score manifest summary": data_root / "raw_manifest_summary.json",
        "RealSR human-test manifest": human_test_manifest / "mos.csv"
        if human_test_manifest.is_dir()
        else human_test_manifest,
    }
    missing = [
        f"{label}: {path}" for label, path in required.items() if not path.is_file()
    ]
    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Missing canonical training input(s):\n  - {details}\nBuild the training dataset with `training_data_construction.py`."
        )
    summary_path = required["raw-score manifest summary"]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            f"Cannot read canonical manifest summary: {summary_path}"
        ) from error
    if summary.get("schema") != "parameter-independent raw score manifest":
        raise RuntimeError(
            f"Unexpected canonical manifest schema in {summary_path}: {summary.get('schema')!r}"
        )
    required_fields = {"reverse_candidate_scores", "reverse_gt_scores"}
    summary_fields = set(summary.get("fields", []))
    if int(summary.get("schema_version", 0)) < 2 or not required_fields.issubset(
        summary_fields
    ):
        raise RuntimeError(
            f"Raw manifest {summary_path} lacks per-Q/A reverse scores required for runtime gate/margin thresholding. Build it with `training_data_construction.py`."
        )
    if require_forward_scores:
        forward_fields = {
            "forward_gt_scores",
            "forward_candidate_scores",
            "forward_n_qa",
        }
        if (
            summary.get("protocol") != "high-low-scored-qa-pooled-v2"
            or int(summary.get("schema_version", 0)) < 5
            or (not forward_fields.issubset(summary_fields))
            or (summary.get("vqa_divisor") != 1)
        ):
            raise RuntimeError(
                "Forward thresholding requires a new scored-Q/A v2 manifest with divisor 1"
            )
    return summary


def compute_target_statistics(
    train_raw_path: Path,
    training_subset: TrainingSubset,
    reverse_gate_threshold: float,
    reverse_margin_threshold: float,
    reverse_weight: float,
    vqa_target_weight: float,
    direct_target_weight: float,
    forward_support_threshold: float | None = None,
) -> TargetStatistics:
    """Scan raw labels only; never writes a parameter-specific dataset."""
    validate_target_options(
        reverse_gate_threshold,
        reverse_margin_threshold,
        vqa_target_weight,
        direct_target_weight,
    )
    vqa_stats = RunningStats()
    direct_stats = RunningStats()
    with train_raw_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not training_subset.matches(row):
                continue
            vqa_value, _ = raw_vqa_target(
                row,
                reverse_gate_threshold,
                reverse_margin_threshold,
                reverse_weight,
                forward_support_threshold,
            )
            vqa_stats.add(vqa_value)
            if direct_target_weight > 0:
                if row["direct_score"] == "":
                    raise RuntimeError(
                        f"Missing direct training score: {row['sample_id']}"
                    )
                direct_stats.add(float(row["direct_score"]))
    if vqa_stats.std <= 0:
        raise RuntimeError(f"Invalid VQA training target std: {vqa_stats.std}")
    if direct_target_weight > 0 and direct_stats.std <= 0:
        raise RuntimeError(f"Invalid direct training target std: {direct_stats.std}")
    mixed_stats = RunningStats()
    with train_raw_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not training_subset.matches(row):
                continue
            vqa_value, _ = raw_vqa_target(
                row,
                reverse_gate_threshold,
                reverse_margin_threshold,
                reverse_weight,
                forward_support_threshold,
            )
            vqa_z = (vqa_value - vqa_stats.mean) / vqa_stats.std
            direct_z = 0.0
            if direct_target_weight > 0:
                direct_z = (
                    float(row["direct_score"]) - direct_stats.mean
                ) / direct_stats.std
            mixed_stats.add(vqa_target_weight * vqa_z + direct_target_weight * direct_z)
    if mixed_stats.std <= 0:
        raise RuntimeError(f"Invalid mixed training target std: {mixed_stats.std}")
    return TargetStatistics(
        vqa_mean=vqa_stats.mean,
        vqa_std=vqa_stats.std,
        direct_mean=direct_stats.mean if direct_target_weight > 0 else None,
        direct_std=direct_stats.std if direct_target_weight > 0 else None,
        mixed_mean=mixed_stats.mean,
        mixed_std=mixed_stats.std,
    )


@dataclass(frozen=True)
class RuntimeTargetMixer:
    """Compute the final training target for both training and validation rows."""

    reverse_gate_threshold: float
    reverse_margin_threshold: float
    reverse_weight: float
    vqa_target_weight: float
    direct_target_weight: float
    statistics: TargetStatistics
    forward_support_threshold: float | None = None

    def __call__(self, source: dict[str, str]) -> dict[str, Any]:
        row: dict[str, Any] = dict(source)
        vqa_value, reverse_penalty = raw_vqa_target(
            row,
            self.reverse_gate_threshold,
            self.reverse_margin_threshold,
            self.reverse_weight,
            self.forward_support_threshold,
        )
        vqa_z = (vqa_value - self.statistics.vqa_mean) / self.statistics.vqa_std
        direct_z: float | str = ""
        if row["direct_score"] != "":
            if (
                self.statistics.direct_mean is not None
                and self.statistics.direct_std is not None
            ):
                direct_z = (
                    float(row["direct_score"]) - self.statistics.direct_mean
                ) / self.statistics.direct_std
        elif self.direct_target_weight > 0:
            raise RuntimeError(
                f"Missing direct score for mixed target: {row['sample_id']}"
            )
        direct_component = float(direct_z) if self.direct_target_weight > 0 else 0.0
        mixed_value = (
            self.vqa_target_weight * vqa_z
            + self.direct_target_weight * direct_component
        )
        target_z = (
            mixed_value - self.statistics.mixed_mean
        ) / self.statistics.mixed_std
        row.update(
            {
                "forward_loss_unthresholded": float(source["forward_loss"]),
                "forward_loss": forward_loss_from_raw_scores(
                    source, self.forward_support_threshold
                ),
                "reverse_penalty": reverse_penalty,
                "vqa_target_raw": vqa_value,
                "vqa_target_z": vqa_z,
                "direct_score_z": direct_z,
                "mixed_target": mixed_value,
                "target_raw": mixed_value,
                "target_z": target_z,
                "target_space": "final_training_target",
            }
        )
        return row

    def summary(self) -> dict[str, Any]:
        return {
            "direction": "higher means more hallucination",
            "vqa_target": f"forward_loss + {self.reverse_weight} * mean_i(1[reverse_candidate_score_i > {self.reverse_gate_threshold}] * reverse_contribution_i)",
            "reverse_contribution": f"max(reverse_margin - {self.reverse_margin_threshold}, 0)"
            if self.reverse_margin_threshold >= 0
            else f"max(reverse_margin, {self.reverse_margin_threshold})",
            "reverse_gate_threshold": self.reverse_gate_threshold,
            "reverse_margin_threshold": self.reverse_margin_threshold,
            "forward_support_threshold": self.forward_support_threshold,
            "forward_contribution": "stored forward_loss"
            if self.forward_support_threshold is None
            else f"mean_all_scored_GT_QA(1[score_gt > {self.forward_support_threshold}] * (score_gt - score_candidate))",
            "direct_score": "raw integer 1-5"
            if self.direct_target_weight > 0
            else None,
            "combined": f"{self.vqa_target_weight} * z(vqa_target) + {self.direct_target_weight} * z(direct_score)",
            "validation_target": "final_training_target",
            "model_target_space": "normalized_final_training_target",
            "mixed_target_mean": self.statistics.mixed_mean,
            "mixed_target_std": self.statistics.mixed_std,
            "normalization": "statistics scanned from train_raw.csv at runtime",
            **self.statistics.as_dict(),
        }


def resolve_training_image(value, manifest_path):
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return str(path)
    roots = (TRAINING_ROOT, manifest_path.parent)
    if not path.is_absolute():
        candidates = [root / path for root in roots]
    else:
        candidates = [
            root.joinpath(*path.parts[index:])
            for root in roots
            for index in range(1, len(path.parts) - 1)
        ]
    matches = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one image under the configured data root for {value}; found {len(matches)}"
        )
    return str(matches.pop())


class IndexedCsvDataset:
    """Random-access CSV with optional runtime row selection."""

    def __init__(
        self,
        path: Path,
        transform: Any = None,
        training_subset: TrainingSubset | None = None,
    ) -> None:
        self.path = path
        self.transform = transform
        self.offsets = array("Q")
        self.total_rows = 0
        image_counts: dict[str, int] = defaultdict(int)
        family_counts: dict[str, int] = defaultdict(int)
        scale_counts: dict[str, int] = defaultdict(int)
        with path.open("rb") as handle:
            header = handle.readline().decode("utf-8").rstrip("\r\n")
            self.fields = next(csv.reader([header]))
            selected_image_ids: list[str] = []
            indices_by_image: dict[str, list[int]] = defaultdict(list)
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                self.total_rows += 1
                if training_subset is not None:
                    values = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))
                    row = dict(zip(self.fields, values))
                    if not training_subset.matches(row):
                        continue
                    image_counts[row["image_id"]] += 1
                    family_counts[row["family"]] += 1
                    scale_counts[row["scale_tag"]] += 1
                    selected_index = len(self.offsets)
                    selected_image_ids.append(row["image_id"])
                    indices_by_image[row["image_id"]].append(selected_index)
                self.offsets.append(offset)
        self.image_id_by_index = tuple(selected_image_ids)
        self.indices_by_image = {
            image_id: tuple(indices) for image_id, indices in indices_by_image.items()
        }
        self.selection_summary: dict[str, Any] | None = None
        if training_subset is not None:
            if not self.offsets:
                raise RuntimeError(f"Training subset selected no rows from {path}")
            observed_pairs = set(image_counts.values())
            expected_pairs = training_subset.expected_pairs_per_gt
            if observed_pairs != {expected_pairs}:
                raise RuntimeError(
                    f"Training subset has inconsistent pairs per GT: expected={expected_pairs}, observed={sorted(observed_pairs)}"
                )
            self.selection_summary = {
                **training_subset.as_dict(),
                "selected_train_samples": len(self.offsets),
                "total_manifest_train_samples": self.total_rows,
                "selected_gt_images": len(image_counts),
                "family_counts": dict(sorted(family_counts.items())),
                "scale_counts": dict(sorted(scale_counts.items())),
            }
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError(f"Expected a pair of training indices, got: {index}")
            return (self[index[0]], self[index[1]])
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        values = next(
            csv.reader([self._handle.readline().decode("utf-8").rstrip("\r\n")])
        )
        row = dict(zip(self.fields, values))
        for field in ("gt_path", "candidate_path"):
            row[field] = resolve_training_image(row[field], self.path)
        return self.transform(row) if self.transform is not None else row


class ImagePairCollator:
    def __init__(self, preprocess: Any, target_field: str) -> None:
        self.preprocess = preprocess
        self.target_field = target_field

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, Any]:
        import torch

        configure_png_metadata_limits()
        gt = []
        candidate = []
        for row in rows:
            with Image.open(row["gt_path"]) as image:
                gt.append(self.preprocess(image.convert("RGB")))
            with Image.open(row["candidate_path"]) as image:
                candidate.append(self.preprocess(image.convert("RGB")))
        return {
            "gt": torch.stack(gt),
            "candidate": torch.stack(candidate),
            "target": torch.tensor(
                [float(row[self.target_field]) for row in rows], dtype=torch.float32
            ),
            "rows": rows,
        }


def seed_data_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: Any,
    collator: Any,
    batch_size: int,
    workers: int,
    seed: int,
    sampler: Any = None,
    shuffle: bool = False,
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collator,
        worker_init_fn=seed_data_worker,
        generator=generator,
        drop_last=sampler is not None,
    )


def evaluation_loader(
    dataset: Any,
    collator: Any,
    batch_size: int,
    workers: int,
    seed: int,
    rank: int,
    world: int,
) -> Any:
    from torch.utils.data import Subset

    return make_loader(
        Subset(dataset, range(rank, len(dataset), world)),
        collator,
        batch_size,
        workers,
        seed,
        shuffle=False,
    )


@dataclass
class TrainingData:
    train_loader: Any
    validation_loader: Any
    human_test_loader: Any
    train_sampler: Any
    train_samples: int
    validation_samples: int
    human_test_samples: int
    training_subset_summary: dict[str, Any]


def build_training_data(
    data_root: Path,
    human_test_manifest: Path,
    training_subset: TrainingSubset,
    target_mixer: RuntimeTargetMixer,
    preprocess: Any,
    batch_size: int,
    workers: int,
    seed: int,
    rank: int,
    world: int,
) -> TrainingData:
    from torch.utils.data import DistributedSampler

    pair_collator = ImagePairCollator(preprocess, "target_z")
    human_collator = ImagePairCollator(preprocess, "human_score")
    train_dataset = IndexedCsvDataset(
        data_root / "train_raw.csv", target_mixer, training_subset=training_subset
    )
    validation_dataset = IndexedCsvDataset(data_root / "val_raw.csv", target_mixer)
    human_dataset = (
        BenchmarkDataset(human_test_manifest)
        if human_test_manifest.is_dir()
        else read_pairs(human_test_manifest)
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed
    )
    train_collator = pair_collator
    return TrainingData(
        train_loader=make_loader(
            train_dataset,
            train_collator,
            batch_size,
            workers,
            seed + rank,
            sampler=train_sampler,
            shuffle=True,
        ),
        validation_loader=evaluation_loader(
            validation_dataset,
            pair_collator,
            batch_size,
            workers,
            seed + 10000 + rank,
            rank,
            world,
        ),
        human_test_loader=evaluation_loader(
            human_dataset,
            human_collator,
            batch_size,
            workers,
            seed + 20000 + rank,
            rank,
            world,
        ),
        train_sampler=train_sampler,
        train_samples=len(train_dataset),
        validation_samples=len(validation_dataset),
        human_test_samples=len(human_dataset),
        training_subset_summary=dict(train_dataset.selection_summary or {}),
    )
