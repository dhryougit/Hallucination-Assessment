"""Evaluate a saved DINO checkpoint on its original human-test manifest."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
import torch


from dataloader import ImagePairCollator, BenchmarkDataset, read_pairs, BENCHMARK_ROOT
from train import DinoV3ClsMetric, load_checkpoint, metric_summary


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--benchmark",
        type=Path,
        default=BENCHMARK_ROOT,
        help="Released benchmark directory",
    )
    source.add_argument("--manifest", type=Path, help="Custom human pair CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    cli = parser.parse_args()
    os.umask(0)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**checkpoint["args"])
    if args.backbone != "dinov3":
        raise ValueError("This evaluator currently supports DINOv3 only")
    torch.manual_seed(args.seed)
    model = DinoV3ClsMetric(args)
    load_checkpoint(model, None, cli.checkpoint, args, require_optimizer=False)
    model = model.cuda().eval()
    manifest = cli.manifest if cli.manifest else cli.benchmark / "mos.csv"
    records = (
        read_pairs(cli.manifest)
        if cli.manifest
        else list(BenchmarkDataset(cli.benchmark))
    )
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Duplicate human-test sample")
    loader = torch.utils.data.DataLoader(
        records,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=ImagePairCollator(model.preprocess, "human_score"),
    )
    reporting = checkpoint.get("reporting_scale") or args.reporting_scale
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    rows = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda", dtype=dtype, enabled=args.precision != "fp32"):
                predicted_z, distance = model(
                    batch["gt"].cuda(), batch["candidate"].cuda()
                )
            prediction = (
                predicted_z.float().cpu().numpy() * reporting["std"] + reporting["mean"]
            )
            for source, pred, raw in zip(
                batch["rows"], prediction, distance.float().cpu().numpy()
            ):
                rows.append(
                    {
                        **source,
                        "target": float(source["human_score"]),
                        "prediction": float(pred),
                        "raw_distance": float(raw),
                    }
                )
            print(f"EVALUATED {len(rows)}/{len(records)}", flush=True)
    if not all(
        np.isfinite(row[key])
        for row in rows
        for key in ("target", "prediction", "raw_distance")
    ):
        raise ValueError("Non-finite score")
    summary = metric_summary(rows, human=True)
    centered_target, centered_prediction = [], []
    for image_id in sorted({row["image_id"] for row in rows}):
        group = [row for row in rows if row["image_id"] == image_id]
        target = np.array([row["target"] for row in group])
        prediction = np.array([row["prediction"] for row in group])
        centered_target.extend(target - target.mean())
        centered_prediction.extend(prediction - prediction.mean())
    summary["image_centered_spearman"] = float(
        spearmanr(centered_target, centered_prediction).statistic
    )
    methods = []
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        methods.append(
            {
                "method": method,
                "pairs": len(group),
                "human_mos": float(np.mean([row["target"] for row in group])),
                "predicted_score": float(np.mean([row["prediction"] for row in group])),
                "cosine_distance": float(
                    np.mean([row["raw_distance"] for row in group])
                ),
            }
        )
    mos = [row["human_mos"] for row in methods]
    pred = [row["predicted_score"] for row in methods]
    for row, human_rank, predicted_rank in zip(methods, rankdata(mos), rankdata(pred)):
        row.update(human_rank=float(human_rank), predicted_rank=float(predicted_rank))
    summary["method_mean_spearman"] = float(spearmanr(mos, pred).statistic)
    summary["method_mean_pearson"] = float(pearsonr(mos, pred).statistic)
    provenance = {
        "checkpoint": str(cli.checkpoint),
        "checkpoint_sha256": digest(cli.checkpoint),
        "step": checkpoint["step"],
        "manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "cls_features": args.cls_features,
        "precision": args.precision,
        "reporting_scale": reporting,
        "score_direction": "higher means more hallucination",
        "method_mean_weighting": "equal weight per image-method pair",
        "model_config": checkpoint.get("model_config"),
        "metrics": summary,
    }
    cli.output.mkdir(parents=True, exist_ok=True)
    write_csv(cli.output / "pair_scores.csv", rows)
    write_csv(cli.output / "method_scores.csv", methods)
    (cli.output / "summary.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"metrics": summary, "methods": methods}, indent=2), flush=True)


if __name__ == "__main__":
    main()
