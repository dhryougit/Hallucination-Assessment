"""Load benchmark image pairs and generated Q/A directly from released data."""
import csv
import json
import math
from pathlib import Path
from collections.abc import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "data/benchmark"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/vha"
METHOD_FOLDERS = {
    'stablesr': 'StableSR', 'diffbir': 'DiffBIR', 'seesr': 'SeeSR',
    'osediff': 'OSEDiff', 'vosr': 'VOSR', 'pure': 'PURE',
}

def read_pairs(path, image_root=None):
    path = Path(path).resolve()
    root = Path(image_root).resolve() if image_root else path.parent
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    keys = set()
    for row in rows:
        # Released MOS files omit paths; retain explicit paths in custom manifests.
        if row.get('image'):
            if row.get('image_id') and row['image_id'] != row['image']:
                raise ValueError('Conflicting image and image_id in pair CSV')
            row['image_id'] = row['image']
            if not row.get('gt_path') or not row.get('candidate_path'):
                method = row.get('method')
                if method not in METHOD_FOLDERS:
                    raise ValueError(f'Unknown benchmark method: {method}')
                row['gt_path'] = row.get('gt_path') or f"images/GT/{row['image']}.png"
                row['candidate_path'] = row.get('candidate_path') or f"images/{METHOD_FOLDERS[method]}/{row['image']}.png"
        for field in ('image_id', 'gt_path', 'candidate_path'):
            if not row.get(field):
                raise ValueError(f'Missing {field} in pair CSV')
        row.setdefault('method', 'sr')
        row.setdefault('sample_id', f"{row['image_id']}__{row['method']}")
        if row['sample_id'] in keys:
            raise ValueError(f"Duplicate pair: {row['sample_id']}")
        keys.add(row['sample_id'])
        for field in ('gt_path', 'candidate_path'):
            p = Path(row[field])
            p = p if p.is_absolute() else root / p
            if not p.is_file():
                raise FileNotFoundError(p)
            row[field] = str(p.resolve())
        if 'human_score' in row and not math.isfinite(float(row['human_score'])):
            raise ValueError('Nonfinite human annotation')
    if not rows:
        raise ValueError('Empty pair CSV')
    return rows


class BenchmarkDataset(Sequence):
    """Image-pair records; image decoding uses the existing model collator."""
    def __init__(self, benchmark):
        self.root = Path(benchmark).resolve()
        self.manifest = self.root / 'mos.csv'
        self.rows = read_pairs(self.manifest, self.root)
        methods = {'diffbir', 'osediff', 'pure', 'seesr', 'stablesr', 'vosr'}
        ids = {row['image_id'] for row in self.rows}
        pairs = {(row['image_id'], row['method']) for row in self.rows}
        if len(self.rows) != 540 or len(ids) != 90 or pairs != {(i, m) for i in ids for m in methods}:
            raise ValueError('Expected the fixed 90-image, six-method, 540-pair benchmark')
        if any(not row.get('human_score') for row in self.rows):
            raise ValueError('Benchmark requires human_score for every pair')

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


def read_qa(path):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (str(Path(row['image_path']).resolve()), row['family'])
            if key in result:
                raise ValueError(f'Duplicate QA record: {key}')
            result[key] = row
    return result
