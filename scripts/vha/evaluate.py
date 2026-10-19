"""Compute VHA for benchmark pairs and compare with human annotations."""
import argparse
from pathlib import Path
from dataloader import BenchmarkDataset, read_pairs, read_qa, BENCHMARK_ROOT, OUTPUT_ROOT
from scoring import score_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--benchmark', type=Path, default=BENCHMARK_ROOT)
    source.add_argument('--pairs', type=Path, help='Optional CSV for your own images')
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--gt-qa', type=Path, default=OUTPUT_ROOT / 'gt_qa.jsonl')
    parser.add_argument('--sr-qa', type=Path, default=OUTPUT_ROOT / 'sr_qa.jsonl')
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'evaluation')
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    pairs = read_pairs(args.pairs, args.image_root) if args.pairs else list(BenchmarkDataset(args.benchmark))
    score_pairs(pairs, read_qa(args.gt_qa), read_qa(args.sr_qa), args)


if __name__ == '__main__':
    main()
