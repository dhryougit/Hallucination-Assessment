#!/usr/bin/env python3
"""Construct the paper VHA+ training data from downloaded LSDIR/TextVQA and PiSA-SR."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from PIL import Image, ImageOps, PngImagePlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCALES = (0.5, 0.375, 0.25)
CONTROLS = (("pix1_sem0", 1.0, 0.0), ("pix0p5_sem1", 0.5, 1.0), ("pix1_sem1", 1.0, 1.0))
PROFILES = {"high": "high-semantic-scene-upto144", "low": "low-level-prompt"}
PngImagePlugin.MAX_TEXT_CHUNK = max(PngImagePlugin.MAX_TEXT_CHUNK, 64 * 1024 * 1024)
PngImagePlugin.MAX_TEXT_MEMORY = max(PngImagePlugin.MAX_TEXT_MEMORY, 256 * 1024 * 1024)


def pisa_presets(pre_upscale):
    prefix = "pisasr_2x" if pre_upscale == 2 else "pisasr"
    return [(prefix + "_" + name, pix, sem) for name, pix, sem in CONTROLS]

def load_model(args: argparse.Namespace) -> Any:
    os.chdir(args.pisa_root)
    sys.path.insert(0, str(args.pisa_root))
    from pisasr import PiSASR_eval
    model_args = SimpleNamespace(pretrained_model_path=str(args.sd_path), pretrained_path=str(args.checkpoint), process_size=512 * args.pre_upscale, upscale=1, align_method='adain', lambda_pix=1.0, lambda_sem=1.0, vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024, latent_tiled_size=96, latent_tiled_overlap=32, mixed_precision='fp16', default=False)
    model = PiSASR_eval(model_args)
    model.set_eval()
    return model

def stable_noise_like(distribution: Any, records: list[dict[str, str]], seed: int) -> torch.Tensor:
    import torch
    chunks = []
    for index, record in enumerate(records):
        token = f"{seed}:{record['image_id']}:{record['scale_tag']}"
        item_seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little')
        generator = torch.Generator(device=distribution.mean.device)
        generator.manual_seed(item_seed)
        chunks.append(torch.randn(distribution.mean[index:index + 1].shape, generator=generator, device=distribution.mean.device, dtype=distribution.mean.dtype))
    noise = torch.cat(chunks, dim=0)
    return distribution.mean + distribution.std * noise

def load_tensor_batch(records: list[dict[str, str]], pre_upscale: int) -> tuple[torch.Tensor, list[Image.Image]]:
    import torch
    from torchvision.transforms.functional import pil_to_tensor
    images = []
    tensors = []
    for record in records:
        image = Image.open(record['lr_image_path']).convert('RGB')
        if image.size != (512, 512):
            raise ValueError(f"Expected 512x512 input: {record['lr_image_path']} -> {image.size}")
        model_input = image.resize((512 * pre_upscale, 512 * pre_upscale), Image.Resampling.BICUBIC)
        images.append(image)
        tensors.append(pil_to_tensor(model_input).float().div_(127.5).sub_(1.0))
    return (torch.stack(tensors).cuda(non_blocking=True), images)

def infer_three_presets(model: Any, records: list[dict[str, str]], seed: int, pre_upscale: int, save_root: Path | None) -> list[dict[str, Any]]:
    from torchvision.transforms.functional import to_pil_image
    inputs, source_images = load_tensor_batch(records, pre_upscale)
    try:
        inputs = inputs.to(dtype=model.weight_dtype)
        source_color = inputs.float().mul(0.5).add(0.5)
        distribution = model.vae.encode(inputs).latent_dist
        encoded = stable_noise_like(distribution, records, seed) * model.vae.config.scaling_factor
        prompt = model.encode_prompt(['']).to(dtype=model.weight_dtype).repeat(len(records), 1, 1)
        semantic_prediction = model.unet(encoded, model.timesteps1, encoder_hidden_states=prompt).sample
        model._apply_ori_weight()
        pixel_prediction = model.unet(encoded, model.timesteps1, encoder_hidden_states=prompt).sample
        model._apply_lora_delta()
        semantic_delta = semantic_prediction - pixel_prediction
        del semantic_prediction
        rows: list[dict[str, Any]] = []
        for variant, lambda_pix, lambda_sem in pisa_presets(pre_upscale):
            prediction = lambda_pix * pixel_prediction + lambda_sem * semantic_delta
            denoised = encoded - prediction
            output = model.vae.decode(denoised / model.vae.config.scaling_factor).sample
            output = output.float().mul(0.5).add(0.5).clamp_(0, 1)
            source_flat = source_color.flatten(2)
            output_flat = output.flatten(2)
            source_mean = source_flat.mean(dim=2, keepdim=True).unsqueeze(-1)
            output_mean = output_flat.mean(dim=2, keepdim=True).unsqueeze(-1)
            source_std = (source_flat.var(dim=2, keepdim=True) + 1e-05).sqrt().unsqueeze(-1)
            output_std = (output_flat.var(dim=2, keepdim=True) + 1e-05).sqrt().unsqueeze(-1)
            output = ((output - output_mean) / output_std * source_std + source_mean).clamp_(0, 1)
            for index, record in enumerate(records):
                if save_root is None:
                    continue
                image = to_pil_image(output[index].float().cpu()).resize((512, 512), Image.Resampling.BICUBIC)
                image_dir = save_root / 'images' / record['image_id']
                image_dir.mkdir(parents=True, exist_ok=True)
                output_path = image_dir / f"{variant}_scale{record['scale_tag']}.png"
                image.save(output_path, compress_level=1)
                rows.append({'image_id': record['image_id'], 'split': record['split'], 'scale': record['scale'], 'scale_tag': record['scale_tag'], 'variant': variant, 'lambda_pix': lambda_pix, 'lambda_sem': lambda_sem, 'lr_image_path': record['lr_image_path'], 'sr_image_path': str(output_path)})
            del prediction, denoised, output
        del pixel_prediction, semantic_delta
        return rows
    finally:
        for image in source_images:
            image.close()
        del inputs

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def selection(args):
    rows = []
    seen = set()
    for dataset in ('lsdir', 'textvqa'):
        for split in ('train', 'val'):
            root = getattr(args, dataset + '_' + split).resolve()
            if not root.is_dir():
                raise FileNotFoundError(root)
            count = args.train_images_per_dataset if split == 'train' else args.validation_images_per_dataset
            if args.limit_per_source:
                count = min(count, args.limit_per_source)
            files = sorted((p for p in root.rglob('*') if p.is_file() and
                            p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}),
                           key=lambda p: p.relative_to(root).as_posix())
            chosen = 0
            for path in files:
                with Image.open(path) as source:
                    w, h = ImageOps.exif_transpose(source).size
                if min(w, h) < 512:
                    continue
                if path.resolve() in seen:
                    raise ValueError(f'Overlapping selected source images across splits: {path}')
                seen.add(path.resolve())
                image_id = f'{dataset}_{split}_{chosen:06d}'
                rows.append({'dataset': dataset, 'split': split, 'image_id': image_id,
                             'source_path': str(path.resolve()), 'source_width': w, 'source_height': h,
                             'gt_path': str(Path(split + '_images') / 'gt' / (image_id + '.png'))})
                chosen += 1
                if chosen == count:
                    break
            if chosen != count and (split == 'train' or chosen == 0):
                raise ValueError(f'{root}: need {count} images at least 512x512, found {chosen}')
    return rows


def eligible_rows(db, rows):
    retained = []
    for row in rows:
        groups = [get(db, 'gt_support_' + family, row['image_id']) for family in PROFILES]
        if any(group is None for group in groups):
            raise ValueError('Run gt-qa, including GT support scoring, before generating/scoring candidates')
        if any(groups):
            retained.append(row)
    return retained


def configuration(args, rows):
    # Data/model paths are part of the resume identity; writing into an input is forbidden.
    root = args.output_root.resolve()
    inputs = [args.lsdir_train, args.lsdir_val, args.textvqa_train, args.textvqa_val,
              args.pisa_root, args.sd_path, args.checkpoint]
    for path in inputs:
        resolved = path.resolve()
        if root == resolved or root in resolved.parents or resolved in root.parents:
            raise ValueError(f'Output and inputs must be separate: {root}, {resolved}')
    if not (args.pisa_root / 'pisasr.py').is_file():
        raise FileNotFoundError(args.pisa_root / 'pisasr.py')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    for name in ('tokenizer', 'text_encoder', 'vae', 'unet', 'scheduler'):
        if not (args.sd_path / name).is_dir():
            raise FileNotFoundError(f'SD-2.1-base diffusers directory is missing {name}: {args.sd_path}')
    return {'version': 2, 'train_images_per_dataset': args.train_images_per_dataset,
            'validation_images_per_dataset': args.validation_images_per_dataset,
            'selection': [{k: row[k] for k in ('dataset','split','image_id','source_path','source_width','source_height')} for row in rows],
            'construction_sha256': sha256(__file__),
            'vha_sources': {str(p.relative_to(PROJECT_ROOT)): sha256(p)
                            for p in sorted((PROJECT_ROOT / 'scripts/vha').rglob('*'))
                            if p.is_file() and p.suffix in ('.py', '.txt')},
            'sources': {name: str(getattr(args, name).resolve())
                        for name in ('lsdir_train', 'lsdir_val', 'textvqa_train', 'textvqa_val')},
            'pisa_root': str(args.pisa_root.resolve()), 'sd_path': str(args.sd_path.resolve()),
            'pisa_source_sha256': sha256(args.pisa_root / 'pisasr.py'),
            'pisa_checkpoint_sha256': sha256(args.checkpoint), 'model': args.model,
            'seed': args.seed, 'scales': SCALES, 'controls': CONTROLS,
            'generation_batch_size': args.generation_batch_size,
            'pisa_batch_size': args.pisa_batch_size,
            'score_batch_size': int(os.environ.get('VQA_SCORE_BATCH_SIZE', '8')),
            'limit_per_source': args.limit_per_source,
            'gt_filter': 'none; all generated GT Q/A retained; score-time gates only',
            'scoring_protocol': {'forward_support_threshold': -2.0, 'reverse_support_threshold': 0.0,
                                 'reverse_margin_threshold': 0.5, 'denominator': 'all_questions'}}


def workspace(args, config):
    root = args.output_root.resolve()
    path = root / 'construction_config.json'
    config = json.loads(json.dumps(config))
    if root.exists() and any(root.iterdir()):
        if not args.resume or not path.is_file():
            raise FileExistsError(f'Use a new output folder, or --resume for this construction: {root}')
        if json.loads(path.read_text()) != config:
            raise ValueError('Resume inputs/configuration changed; use a new output folder')
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, config)
    db = sqlite3.connect(root / 'construction.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind,id))')
    return root, db


def get(db, kind, key):
    row = db.execute('SELECT payload FROM records WHERE kind=? AND id=?', (kind, key)).fetchone()
    return json.loads(row[0]) if row else None


def put(db, kind, key, value):
    db.execute('INSERT INTO records VALUES (?,?,?)', (kind, key, json.dumps(value, ensure_ascii=False, allow_nan=False)))
    db.commit()


def degraded(image, scale):
    w, h = image.size
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BICUBIC).resize((w, h), Image.Resampling.BICUBIC)


def crops(root, db, rows):
    for index, row in enumerate(rows):
        path = root / row['gt_path']
        source_hash = sha256(row['source_path'])
        previous = get(db, 'crop', row['image_id'])
        if previous:
            if source_hash != previous['source_sha256'] or not path.is_file() or sha256(path) != previous['gt_sha256']:
                raise ValueError(f'Crop/source changed: {row["image_id"]}')
            continue
        with Image.open(row['source_path']) as source:
            source = ImageOps.exif_transpose(source).convert('RGB')
            w, h = source.size
            if (w, h) != (int(row['source_width']), int(row['source_height'])) or min(w, h) < 512:
                raise ValueError(f'Selected source dimensions changed: {row["source_path"]}, {(w,h)}')
            left, top = (w - 512) // 2, (h - 512) // 2
            crop = source.crop((left, top, left + 512, top + 512))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp');crop.save(tmp, format='PNG', compress_level=3)
            if path.exists() and sha256(path) != sha256(tmp):
                tmp.unlink();raise ValueError(f'Unrecorded crop differs from the selected source: {path}')
            tmp.replace(path)
        put(db, 'crop', row['image_id'], {'source_sha256': source_hash, 'gt_sha256': sha256(path)})
        if (index + 1) % 100 == 0:
            print('CROPS', index + 1, len(rows), flush=True)


def vha_runtime(args):
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts/vha'))
    import qa_generation
    import scoring
    import transformers
    if transformers.__version__ != '4.57.6':
        raise RuntimeError('VHA requires requirements.txt (Transformers 4.57.6)')
    os.environ['QA_GENERATION_DO_SAMPLE'] = '0'
    os.environ['QA_GENERATION_SEED'] = str(args.seed)
    core = qa_generation.model_core();core.seed_generation()
    model, processor = core.load_model_and_processor(args.model, args.device)
    return qa_generation, scoring, core, model, processor


def generate_record(runtime, image, image_id, family, args):
    generation, scoring, core, model, processor = runtime
    generation._low_patch_core(core, PROFILES[family], 'fixed')
    return generation.generate_questions(core, model, processor, image, image_id,
                                         args.device, args.generation_batch_size)


def gt_qa(root, db, rows, args):
    runtime = vha_runtime(args)
    for index, row in enumerate(rows):
        image_id = row['image_id']
        crop = get(db, 'crop', image_id)
        path = root / row['gt_path']
        if not crop or sha256(path) != crop['gt_sha256']:
            raise ValueError(f'Run crops first; missing/changed crop: {image_id}')
        with Image.open(path) as source:
            image = source.convert('RGB')
        for family in PROFILES:
            if get(db, 'gt_qa_' + family, image_id) is None:
                put(db, 'gt_qa_' + family, image_id, generate_record(runtime, image, image_id, family, args))
            if get(db, 'gt_support_' + family, image_id) is None:
                questions = get(db, 'gt_qa_' + family, image_id)['questions']
                values = runtime[1].score_image_questions(runtime[3], runtime[4], runtime[2],
                                                         questions, [image], args.device)
                put(db, 'gt_support_' + family, image_id, values)
        if (index + 1) % 25 == 0:
            print('GT_QA', index + 1, len(rows), flush=True)

def candidates(rows):
    for row in rows:
        for pre in (1, 2):
            family = 'pisasr' if pre == 1 else 'pisasr_2x'
            for scale in SCALES:
                tag = str(scale).replace('.', 'p')
                for variant, pix, sem in pisa_presets(pre):
                    yield {**row, 'candidate_id': f'{family}-{variant}-s{tag}__{row["image_id"]}',
                           'family': family, 'variant': variant, 'scale': scale, 'scale_tag': tag,
                           'pre_upscale': pre,
                           'candidate_path': str(Path(row['split'] + '_images') / family / 'images' /
                                                 row['image_id'] / f'{variant}_scale{tag}.png')}


def sr(root, db, rows, args):
    import torch
    rows = eligible_rows(db, rows)
    # The original generator shares latent noise and UNet predictions across three controls.
    for pre in (1, 2):
        args.pre_upscale = pre
        model = None
        groups = {}
        for item in candidates(rows):
            if item['pre_upscale'] == pre:
                groups.setdefault((item['split'], item['image_id'], item['scale_tag']), []).append(item)
        pending = []
        def run_batch(batch):
            nonlocal model
            if model is None:
                model = load_model(args)
            records = []
            for items in batch:
                sample = items[0]
                lr = root / 'lr_images' / sample['image_id'] / ('lr_s' + sample['scale_tag'] + '.png')
                lr.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(root / sample['gt_path']) as gt:
                    degraded(gt.convert('RGB'), sample['scale']).save(lr)
                records.append({'image_id': sample['image_id'], 'split': sample['split'],
                                'scale': sample['scale'], 'scale_tag': sample['scale_tag'], 'lr_image_path': str(lr)})
            # Batch only within one split so the original save layout is preserved.
            save_root = root / (batch[0][0]['split'] + '_images') / batch[0][0]['family']
            with torch.inference_mode():
                generated = infer_three_presets(model, records, args.seed, pre, save_root)
            expected = {str(root / item['candidate_path']) for items in batch for item in items}
            if {item['sr_image_path'] for item in generated} != expected:
                raise ValueError('PiSA output coverage mismatch')
            for items in batch:
                for item in items:
                    key = item['candidate_id'];path = root / item['candidate_path']
                    if get(db, 'sr', key) is None:
                        db.execute('INSERT INTO records VALUES (?,?,?)',
                                   ('sr', key, json.dumps({'sha256': sha256(path)})))
                db.commit()
                print('SR', items[0]['image_id'], items[0]['scale_tag'], items[0]['family'], flush=True)
        for items in groups.values():
            completed = [get(db, 'sr', x['candidate_id']) for x in items]
            for item, old in zip(items, completed):
                if old and (not (root / item['candidate_path']).is_file() or sha256(root / item['candidate_path']) != old['sha256']):
                    raise ValueError(f'SR image changed: {item["candidate_id"]}')
            if all(completed):
                continue
            if any(completed):
                # Resume a partially committed preset group without changing recorded images.
                raise ValueError('Partial PiSA preset group: inspect the construction database before resuming')
            if pending and (pending[0][0]['split'] != items[0]['split'] or len(pending) == args.pisa_batch_size):
                run_batch(pending);pending = []
            pending.append(items)
        if pending:
            run_batch(pending)
        del model
        torch.cuda.empty_cache()


def sr_qa(root, db, rows, args):
    rows = eligible_rows(db, rows)
    runtime = vha_runtime(args)
    for index, item in enumerate(candidates(rows)):
        key = item['candidate_id'];path = root / item['candidate_path'];stored = get(db, 'sr', key)
        if not stored or sha256(path) != stored['sha256']:
            raise ValueError(f'Run sr first; missing/changed SR image: {key}')
        with Image.open(path) as source:
            image = source.convert('RGB')
        for family in PROFILES:
            if get(db, 'sr_qa_' + family, key) is None:
                put(db, 'sr_qa_' + family, key, generate_record(runtime, image, key, family, args))
        if (index + 1) % 25 == 0:
            print('SR_QA', index + 1, flush=True)


def raw_manifest_row(item, gt, reverse):
    hf, lf = (gt[f] for f in PROFILES)
    hr, lr = (reverse[f] for f in PROFILES)
    forwards = hf + lf;backwards = hr + lr
    nf, nr = len(forwards), len(backwards)
    if nf == 0:
        raise ValueError(f'No generated GT Q/A: {item["candidate_id"]}')
    fg = [q['scores'][0] for q in forwards];fc = [q['scores'][1] for q in forwards]
    rg = [q['scores'][0] for q in backwards];rc = [q['scores'][1] for q in backwards]
    if not all(math.isfinite(v) for array in (fg,fc,rg,rc) for v in array):
        raise ValueError('Nonfinite VHA training scores')
    return {'sample_id': item['candidate_id'], 'image_id': item['image_id'],
            'dataset': 'LSDIR' if item['dataset'] == 'lsdir' else 'TextVQA', 'split': item['split'],
            'family': item['family'], 'variant': item['variant'], 'scale': item['scale'], 'scale_tag': item['scale_tag'],
            'gt_path': item['gt_path'], 'candidate_path': item['candidate_path'],
            'forward_gt_score': sum(fg) / nf, 'forward_candidate_score': sum(fc) / nf,
            'forward_loss': sum(fg) / nf - sum(fc) / nf, 'forward_n_qa': nf,
            'forward_gt_scores': json.dumps(fg), 'forward_candidate_scores': json.dumps(fc),
            'reverse_gt_scores': json.dumps(rg), 'reverse_candidate_scores': json.dumps(rc),
            'reverse_gt_score': sum(rg) / nr if nr else '', 'reverse_candidate_score': sum(rc) / nr if nr else '',
            'reverse_margin': (sum(rc) - sum(rg)) / nr if nr else '',
            'reverse_n_qa': nr, 'reverse_available': int(nr > 0), 'direct_score': '', 'vqa_divisor': 1,
            'high_forward_n': len(hf), 'low_forward_n': len(lf), 'high_reverse_n': len(hr), 'low_reverse_n': len(lr)}


def scoring_stage(root, db, rows, args):
    rows = eligible_rows(db, rows)
    runtime = vha_runtime(args)
    for index, item in enumerate(candidates(rows)):
        key = item['candidate_id']
        if get(db, 'score', key) is not None:
            continue
        with Image.open(root / item['gt_path']) as gt, Image.open(root / item['candidate_path']) as sr_image:
            images = [gt.convert('RGB'), sr_image.convert('RGB')]
        forwards, reverse = {}, {}
        for family in PROFILES:
            retained = get(db, 'gt_support_' + family, item['image_id'])
            generated = get(db, 'sr_qa_' + family, key)
            if retained is None or generated is None:
                raise ValueError(f'Run GT support scoring and SR Q/A generation before scoring: {key}/{family}')
            forwards[family] = runtime[1].score_image_questions(runtime[3], runtime[4], runtime[2], retained, images, args.device)
            # Preserve the original cached GT support and text prior for the forward scores.
            for old, current in zip(retained, forwards[family]):
                current['scores'] = [old['scores'][0], current['image_logprobs'][1] - old['text_prior']]
                current['text_prior'] = old['text_prior']
            reverse[family] = runtime[1].score_image_questions(runtime[3], runtime[4], runtime[2], generated['questions'], images, args.device)
        put(db, 'score', key, {'manifest': raw_manifest_row(item, forwards, reverse),
                             'forward': forwards, 'reverse': reverse})
        if (index + 1) % 25 == 0:
            print('VHA_SCORES', index + 1, flush=True)


def manifest(root, db, rows):
    rows = eligible_rows(db, rows)
    target = root / 'manifests';target.mkdir(exist_ok=True)
    fields = ['sample_id','image_id','dataset','split','family','variant','scale','scale_tag','gt_path','candidate_path',
              'forward_gt_score','forward_candidate_score','forward_loss','forward_n_qa','reverse_candidate_scores',
              'reverse_gt_scores','reverse_candidate_score','reverse_gt_score','reverse_margin','reverse_n_qa',
              'reverse_available','direct_score','vqa_divisor','high_forward_n','low_forward_n','high_reverse_n',
              'low_reverse_n','forward_candidate_scores','forward_gt_scores']
    counts = Counter();families = {'train': Counter(), 'val': Counter()}
    handles = {s: (target / (s + '_raw.csv.tmp')).open('w', newline='') for s in ('train','val')}
    writers = {s: csv.DictWriter(h, fieldnames=fields, lineterminator='\n') for s,h in handles.items()}
    for writer in writers.values():
        writer.writeheader()
    try:
        for item in candidates(rows):
            score = get(db, 'score', item['candidate_id'])
            if score is None:
                raise ValueError(f'Incomplete VHA scores: {item["candidate_id"]}')
            row = dict(score['manifest'])
            for field in ('gt_path', 'candidate_path'):
                row[field] = str(Path('..') / row[field])
            writers[item['split']].writerow(row)
            counts[item['split']] += 1;families[item['split']][item['family']] += 1
        if not all(counts[s] for s in handles):
            raise ValueError('Both train and validation splits must be nonempty')
    finally:
        for handle in handles.values():handle.close()
    for split in handles:
        (target / (split + '_raw.csv.tmp')).replace(target / (split + '_raw.csv'))
    summary = {'schema': 'parameter-independent raw score manifest', 'schema_version': 5,
               'protocol': 'high-low-scored-qa-pooled-v2', 'fields': fields, 'vqa_divisor': 1,
               'splits': dict(counts), 'family_counts': {k:dict(v) for k,v in families.items()},
               'forward_pooling': 'all generated High and Low GT Q/As; score-time gates retain all questions in the denominator',
               'reverse_pooling': 'all generated High and Low candidate Q/As, including zero-masked contributions',
               'qa_prefilter': 'none', 'scoring_protocol': {'forward_support_threshold': -2.0,
               'reverse_support_threshold': 0.0, 'reverse_margin_threshold': 0.5,
               'denominator': 'all_questions'}, 'profiles': PROFILES, 'direct_score': 'not used', 'construction_config': '../construction_config.json'}
    write_json(target / 'raw_manifest_summary.json', summary)
    print('TRAINING_DATA_COMPLETE', dict(counts), target, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lsdir-train','lsdir-val','textvqa-train','textvqa-val','pisa-root','sd-path','checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'data/training')
    parser.add_argument('--pisa-python', default=sys.executable, help='Python executable in the official PiSA-SR environment')
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pisa-batch-size', type=int, default=1)
    parser.add_argument('--generation-batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', choices=('all','crops','gt-qa','sr','sr-qa','scoring','manifest'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--train-images-per-dataset', type=int, default=20000)
    parser.add_argument('--validation-images-per-dataset', type=int, default=250)
    parser.add_argument('--limit-per-source', type=int, help='Small smoke-test subset; omit for the paper split')
    args = parser.parse_args()
    for name in ('lsdir_train','lsdir_val','textvqa_train','textvqa_val','pisa_root','sd_path','checkpoint','output_root'):
        setattr(args, name, getattr(args, name).resolve())
    if min(args.pisa_batch_size, args.generation_batch_size, args.train_images_per_dataset, args.validation_images_per_dataset) < 1 or (args.limit_per_source is not None and args.limit_per_source < 1):
        parser.error('Batch sizes and subset limit must be positive')
    return args


def main():
    args = parse_args();os.umask(0)
    rows = selection(args);config = configuration(args, rows)
    if args.dry_run:
        print(json.dumps({'source_images': dict(Counter(r['dataset'] + '_' + r['split'] for r in rows)),
                          'candidate_count_note': 'Maximum before removing images with no retained GT Q/A',
                          'maximum_candidate_pairs': dict(Counter(item['split'] for item in candidates(rows))),
                          'stages': ['crops','gt-qa','sr','sr-qa','scoring','manifest'],
                          'output_root': str(args.output_root)}, indent=2))
        return
    root, db = workspace(args, config)
    try:
        if args.stage == 'all':
            db.close()
            for stage in ('crops','gt-qa','sr','sr-qa','scoring','manifest'):
                python = args.pisa_python if stage == 'sr' else sys.executable
                command = [python, str(Path(__file__).resolve()), *sys.argv[1:], '--stage', stage, '--resume']
                print('CONSTRUCTION_STAGE', stage, flush=True)
                subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        elif args.stage == 'crops':crops(root, db, rows)
        elif args.stage == 'gt-qa':gt_qa(root, db, rows, args)
        elif args.stage == 'sr':sr(root, db, rows, args)
        elif args.stage == 'sr-qa':sr_qa(root, db, rows, args)
        elif args.stage == 'scoring':scoring_stage(root, db, rows, args)
        else:manifest(root, db, rows)
    finally:
        db.close()


if __name__ == '__main__':
    main()
