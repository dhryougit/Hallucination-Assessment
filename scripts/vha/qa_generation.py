"""Generate fixed High/Low Q/A using the four templates in prompts/."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import importlib.util
import statistics
from collections import defaultdict
from types import ModuleType, SimpleNamespace
from PIL import Image
from scipy.stats import pearsonr, spearmanr
import hashlib
from types import SimpleNamespace
from pathlib import Path
from dataloader import BenchmarkDataset, BENCHMARK_ROOT, OUTPUT_ROOT
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_high_TAXONOMIES = ['text_logo', 'local_detail', 'object_semantic', 'background_semantic']

_high_DEFINITIONS = {'text_logo': 'Readable text, words, numbers, signs, brands, or logos.', 'local_detail': 'A small visual property or structure inside an object or local region, such as texture, pattern, shape, style, material, decoration, repeated elements, or a visible object part.', 'object_semantic': 'A distinct medium or large physical object or structure, such as a tree, grass, tower, wall, house, vehicle, bridge, or signboard.', 'background_semantic': 'The overall scene, environment, or image-level context, such as a street, village, station, storefront scene, industrial area, or natural landscape. Do not use this taxonomy for a single background object.'}

_high_CAPS = {'text_logo': 4, 'local_detail': 3, 'object_semantic': 2, 'background_semantic': 1}

_high_SCENE_HIGH_TAXONOMIES = ('scene_semantic', 'object_semantic', 'text_logo')

_high_SCENE_HIGH_DEFINITIONS = {'scene_semantic': 'The overall scene, environment, or image-level context. This taxonomy must describe the image as a whole or a broad scene region, not a single object.', 'object_semantic': 'A distinct object or structure that contributes meaningfully to the image content, such as a tree, grass, tower, wall, house, vehicle, bridge, or signboard; or a recognizable visual property or part within an object or local region, such as texture, style, material, decoration, or a visible object part.', 'text_logo': 'Clearly readable text, words, numbers, signs, brands, or logos.'}

_high_HIGH_PROMPT_PROFILES = {'high-semantic-scene-upto144': {'taxonomies': _high_SCENE_HIGH_TAXONOMIES, 'caps': {'scene_semantic': 1, 'object_semantic': 4, 'text_logo': 4}, 'definitions': _high_SCENE_HIGH_DEFINITIONS, 'family': 'high', 'high_taxonomies': _high_SCENE_HIGH_TAXONOMIES}}

_high_CURRENT_HIGH_EVIDENCE_PROMPT = (PROMPT_DIR / 'high_evidence.txt').read_text(encoding='utf-8')

_high_CURRENT_HIGH_QA_PROMPT = (PROMPT_DIR / 'high_qa.txt').read_text(encoding='utf-8')

def _high_high_profile_spec(profile_name: str) -> dict[str, Any]:
    try:
        return _high_HIGH_PROMPT_PROFILES[profile_name]
    except KeyError as error:
        raise ValueError(f'Unknown High-level prompt profile: {profile_name}') from error

def _high_build_high_inventory_prompt(image_id: str, profile_name: str) -> str:
    if profile_name != 'high-semantic-scene-upto144':
        raise ValueError(f'Unknown High-level prompt profile: {profile_name}')
    return _high_CURRENT_HIGH_EVIDENCE_PROMPT.replace('{image_id}', image_id)

def _high_build_high_qa_prompt(image_id: str, taxonomy: str, evidence_rows: list[dict[str, str]], profile_name: str) -> str:
    spec = _high_high_profile_spec(profile_name)
    items = []
    for row in evidence_rows:
        metadata = json.loads(row['scope_rationale'])
        items.append({'evidence_id': row['evidence_id'], 'taxonomy': taxonomy, 'region_scope': metadata['region_scope'], 'region_hint': row['evidence_hint'], 'target': row['target'], 'hallucination_risk': metadata['hallucination_risk']})
    template = _high_CURRENT_HIGH_QA_PROMPT
    prompt = template.replace('{image_id}', image_id)
    prompt = prompt.replace('{taxonomy}', taxonomy)
    prompt = prompt.replace('{taxonomy_definition}', spec['definitions'][taxonomy])
    prompt = prompt.replace('{taxonomy_cap}', str(spec['caps'][taxonomy]))
    prompt = prompt.replace('{evidence_json}', json.dumps(items, indent=2, ensure_ascii=False))
    return prompt.replace('{example_evidence_id}', evidence_rows[0]['evidence_id'])

def _high_load_model_and_processor(model_path: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoConfig, AutoProcessor
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.model_type == 'qwen3_5':
        from transformers import AutoModelForMultimodalLM
        model_cls = AutoModelForMultimodalLM
    else:
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model_cls = Qwen3VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dtype_arg = {'dtype': torch.bfloat16} if config.model_type == 'qwen3_5' else {'torch_dtype': torch.bfloat16}
    model = model_cls.from_pretrained(model_path, device_map=device, trust_remote_code=True, **dtype_arg)
    model.eval()
    return (model, processor)

def _high_apply_generation_template(model: Any, processor: Any, messages: list[dict[str, Any]]) -> str:
    kwargs: dict[str, Any] = {'tokenize': False, 'add_generation_prompt': True}
    if getattr(model.config, 'model_type', '') == 'qwen3_5':
        kwargs['enable_thinking'] = False
    return processor.apply_chat_template(messages, **kwargs)

def _high_generation_kwargs(max_new_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {'max_new_tokens': max_new_tokens}
    if os.environ.get('QA_GENERATION_DO_SAMPLE', '0') == '1':
        kwargs.update({'do_sample': True, 'temperature': float(os.environ.get('QA_GENERATION_TEMPERATURE', '0.8')), 'top_p': float(os.environ.get('QA_GENERATION_TOP_P', '0.95'))})
    else:
        kwargs['do_sample'] = False
    return kwargs

def _high_seed_generation() -> None:
    seed_text = os.environ.get('QA_GENERATION_SEED')
    if seed_text is None:
        return
    import torch
    seed = int(seed_text)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _high_build_messages(question_or_prompt: str, image: Any | None, for_scoring: bool=False) -> list[dict[str, Any]]:
    text = question_or_prompt if not for_scoring else f'Answer the question with the exact short answer only.\nQuestion: {question_or_prompt}\nAnswer:'
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({'type': 'image', 'image': image})
    content.append({'type': 'text', 'text': text})
    return [{'role': 'user', 'content': content}]

def _high_generate_text(model: Any, processor: Any, image: Any, prompt: str, device: str, max_new_tokens: int) -> str:
    import torch
    _high_seed_generation()
    messages = _high_build_messages(prompt, image=image, for_scoring=False)
    text = _high_apply_generation_template(model, processor, messages)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to(device)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
        generated = model.generate(**inputs, **_high_generation_kwargs(max_new_tokens))
    prompt_len = inputs['input_ids'].shape[1]
    return processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]

def _high_generate_text_batch(model: Any, processor: Any, images: list[Any], prompts: list[str], device: str, max_new_tokens: int) -> list[str]:
    """Generate one response per image/prompt pair with OOM-safe batch splitting."""
    import time
    import torch
    if len(images) != len(prompts):
        raise ValueError('images and prompts must have the same length')
    if not images:
        return []
    _high_seed_generation()
    if len(images) == 1:
        return [_high_generate_text(model, processor, images[0], prompts[0], device, max_new_tokens)]
    messages = [_high_build_messages(prompt, image=image, for_scoring=False) for image, prompt in zip(images, prompts)]
    texts = [_high_apply_generation_template(model, processor, message) for message in messages]
    old_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = 'left'
    try:
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        started_at = time.monotonic()
        inputs = processor(text=texts, images=images, padding=True, return_tensors='pt').to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            generated = model.generate(**inputs, **_high_generation_kwargs(max_new_tokens))
        prompt_len = inputs['input_ids'].shape[1]
        decoded = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            print('GENERATION_BATCH_MEMORY', f'batch={len(images)}', f'peak_allocated_gib={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}', f'peak_reserved_gib={torch.cuda.max_memory_reserved() / 2 ** 30:.2f}', f'elapsed_seconds={time.monotonic() - started_at:.2f}', flush=True)
        return decoded
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if 'out of memory' not in str(error).lower():
            raise
        if 'generated' in locals():
            del generated
        if 'inputs' in locals():
            del inputs
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        midpoint = len(images) // 2
        print('GENERATION_BATCH_OOM_SPLIT', f'batch={len(images)}', f'left={midpoint}', f'right={len(images) - midpoint}', flush=True)
        return _high_generate_text_batch(model, processor, images[:midpoint], prompts[:midpoint], device, max_new_tokens) + _high_generate_text_batch(model, processor, images[midpoint:], prompts[midpoint:], device, max_new_tokens)
    finally:
        processor.tokenizer.padding_side = old_padding_side

def _high_mean_answer_log_prob(model: Any, processor: Any, messages: list[dict[str, Any]], image: Any | None, answer: str, device: str) -> float:
    import torch
    prefix = _high_apply_generation_template(model, processor, messages)
    images = None if image is None else [image]
    prefix_enc = processor(text=[prefix], images=images, return_tensors='pt').to(device)
    full_enc = processor(text=[prefix + answer], images=images, return_tensors='pt').to(device)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
        logits = model(**full_enc).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    prefix_count = int(prefix_enc['attention_mask'][0].sum().item())
    full_mask = full_enc['attention_mask'][0]
    active = torch.nonzero(full_mask, as_tuple=False).squeeze(1)
    full_start = int(active[0].item())
    full_end = int(active[-1].item()) + 1
    answer_start = full_start + prefix_count
    answer_ids = full_enc['input_ids'][0, answer_start:full_end]
    if answer_ids.numel() == 0:
        return -1000000000.0
    answer_log_probs = log_probs[0, answer_start - 1:full_end - 1]
    token_log_probs = answer_log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
    return float(token_log_probs.mean().item())

_low_LOW_LEVEL_TAXONOMIES = ('color_tone', 'edge_shape', 'texture')

_low_PROMPT_PROFILES = {'low-level-prompt': {'taxonomies': _low_LOW_LEVEL_TAXONOMIES, 'cap': 3, 'family': 'low', 'high_taxonomies': ('text_logo', 'object_semantic', 'background_semantic')}}

_low_HIGH_PROMPT_PROFILES = ('high-semantic-scene-upto144',)

_low_DEFAULT_PROMPT_PROFILE = 'low-level-prompt'

_low_LOW_LEVEL_EVIDENCE_PROMPT = (PROMPT_DIR / 'low_evidence.txt').read_text(encoding='utf-8')

_low_LOW_LEVEL_QA_PROMPT = (PROMPT_DIR / 'low_qa.txt').read_text(encoding='utf-8')

def _low_load_high_level_prompts() -> ModuleType:
    return model_core()

def _low_profile_spec(profile_name: str, high_module: ModuleType | None=None) -> dict[str, Any]:
    if profile_name == _low_DEFAULT_PROMPT_PROFILE:
        return _low_PROMPT_PROFILES[profile_name]
    module = high_module or _low_load_high_level_prompts()
    return module.high_profile_spec(profile_name)

def _low_profile_caps(profile_name: str, high_module: ModuleType | None=None) -> dict[str, int]:
    spec = _low_profile_spec(profile_name, high_module)
    if 'caps' in spec:
        return dict(spec['caps'])
    return {taxonomy: int(spec['cap']) for taxonomy in spec['taxonomies']}

def _low_is_high_profile(profile_name: str) -> bool:
    return profile_name in _low_HIGH_PROMPT_PROFILES

def _low_evidence_prompt(image_id: str, profile_name: str=_low_DEFAULT_PROMPT_PROFILE, high_module: ModuleType | None=None) -> str:
    if profile_name == _low_DEFAULT_PROMPT_PROFILE:
        return _low_LOW_LEVEL_EVIDENCE_PROMPT.replace('{image_id}', image_id)
    module = high_module or _low_load_high_level_prompts()
    return module.build_high_inventory_prompt(image_id, profile_name)

def _low_schema_aligned_prompt_items(taxonomy: str, evidence_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    items = []
    for row in evidence_rows:
        metadata = json.loads(row['scope_rationale'])
        items.append({'evidence_id': row['evidence_id'], 'taxonomy': taxonomy, 'region_scope': metadata['region_scope'], 'region_hint': row['evidence_hint'], 'target': row['target'], 'hallucination_risk': metadata['hallucination_risk']})
    return items

def _low_qa_prompt(image_id: str, taxonomy: str, evidence_rows: list[dict[str, str]], prompt_style: str='fixed', profile_name: str=_low_DEFAULT_PROMPT_PROFILE, high_module: ModuleType | None=None) -> str:
    if profile_name != _low_DEFAULT_PROMPT_PROFILE:
        module = high_module or _low_load_high_level_prompts()
        return module.build_high_qa_prompt(image_id, taxonomy, evidence_rows, profile_name)
    evidence_json = json.dumps(_low_schema_aligned_prompt_items(taxonomy, evidence_rows), indent=2, ensure_ascii=False)
    prompt = _low_LOW_LEVEL_QA_PROMPT.replace('{image_id}', image_id)
    return prompt.replace('{evidence_json}', evidence_json)

def _low_prompt_image_id(internal_id: str) -> str:
    return internal_id.split('__', 1)[-1]

def _low_normalize_inventory(raw_items: list[dict[str, Any]], image_id: str, taxonomies: tuple[str, ...], caps: dict[str, int], require_answer: bool=False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str, str]] = set()
    for item in raw_items:
        taxonomy = str(item.get('taxonomy', '')).strip()
        target = str(item.get('target', '')).strip()
        region_hint = str(item.get('region_hint', '')).strip()
        region_scope = str(item.get('region_scope', '')).strip()
        risk = str(item.get('hallucination_risk', '')).strip()
        answer = str(item.get('answer', '')).strip()
        if taxonomy not in taxonomies or not target or (not region_hint):
            continue
        if require_answer and (not 1 <= len(answer.split()) <= 3):
            continue
        key = (taxonomy, target.casefold(), region_hint.casefold())
        if key in seen or counts[taxonomy] >= caps[taxonomy]:
            continue
        seen.add(key)
        counts[taxonomy] += 1
        row = {'evidence_id': f'le{len(rows) + 1:02d}', 'image_id': image_id, 'taxonomy': taxonomy, 'target': target, 'evidence_hint': region_hint, 'scope_rationale': json.dumps({'region_scope': region_scope, 'hallucination_risk': risk}, ensure_ascii=False, sort_keys=True)}
        if require_answer:
            row['answer'] = answer
        rows.append(row)
    return rows

def _low_normalize_high_inventory(raw_items: list[dict[str, Any]], image_id: str, taxonomies: tuple[str, ...], caps: dict[str, int], require_answer: bool=False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str, str]] = set()
    for item in raw_items:
        taxonomy = str(item.get('taxonomy', '')).strip()
        target = str(item.get('target', '')).strip()
        evidence_hint = str(item.get('evidence_hint', '')).strip()
        scope_rationale = str(item.get('scope_rationale', '')).strip()
        answer = str(item.get('answer', '')).strip()
        if taxonomy not in taxonomies or not target or (not evidence_hint):
            continue
        if require_answer and (not 1 <= len(answer.split()) <= 3):
            continue
        key = (taxonomy, target.casefold(), evidence_hint.casefold())
        if key in seen or counts[taxonomy] >= caps[taxonomy]:
            continue
        seen.add(key)
        counts[taxonomy] += 1
        row = {'evidence_id': f'e{len(rows) + 1:02d}', 'image_id': image_id, 'taxonomy': taxonomy, 'target': target, 'evidence_hint': evidence_hint, 'scope_rationale': scope_rationale}
        if require_answer:
            row['answer'] = answer
        rows.append(row)
    return rows

def _low_normalize_schema_aligned_high_inventory(raw_items: list[dict[str, Any]], image_id: str, taxonomies: tuple[str, ...], caps: dict[str, int]) -> list[dict[str, str]]:
    rows = _low_normalize_inventory(raw_items, image_id, taxonomies, caps)
    for index, row in enumerate(rows, start=1):
        row['evidence_id'] = f'he{index:02d}'
    return rows

def _low_normalize_qas(raw_items: list[dict[str, Any]], image_id: str, taxonomy: str, evidence_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    evidence_by_id = {row['evidence_id']: row for row in evidence_rows}
    rows: list[dict[str, str]] = []
    used: set[str] = set()
    for item in raw_items:
        evidence_id = str(item.get('evidence_id', '')).strip()
        question = str(item.get('question', item.get('q', ''))).strip()
        answer = str(item.get('answer', item.get('a', ''))).strip()
        if evidence_id not in evidence_by_id or evidence_id in used or (not question) or (not answer):
            continue
        used.add(evidence_id)
        if not question.endswith('?'):
            question += '?'
        evidence = evidence_by_id[evidence_id]
        rows.append({'candidate_id': f'{image_id}_{evidence_id}', 'evidence_id': evidence_id, 'image_id': image_id, 'question': question, 'answer': answer, 'taxonomy': taxonomy, 'answer_type': 'low_level_property', 'target': evidence['target'], 'evidence_hint': evidence['evidence_hint'], 'hr_support': 'yes', 'language_prior_risk': 'low', 'too_tiny_risk': 'low', 'low_detail_risk': 'low'})
    return rows

def _low_normalize_high_qas(raw_items: list[dict[str, Any]], image_id: str, taxonomy: str, evidence_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    evidence_by_id = {row['evidence_id']: row for row in evidence_rows}
    rows: list[dict[str, str]] = []
    used: set[str] = set()
    for item in raw_items:
        evidence_id = str(item.get('evidence_id', '')).strip()
        question = str(item.get('question', item.get('q', ''))).strip()
        answer = str(item.get('answer', item.get('a', ''))).strip()
        if evidence_id not in evidence_by_id or evidence_id in used or (not question) or (not answer):
            continue
        used.add(evidence_id)
        if not question.endswith('?'):
            question += '?'
        evidence = evidence_by_id[evidence_id]
        rows.append({'candidate_id': f'{image_id}_{evidence_id}', 'evidence_id': evidence_id, 'image_id': image_id, 'question': question, 'answer': answer, 'taxonomy': taxonomy, 'answer_type': 'text_or_logo' if taxonomy == 'text_logo' else 'scene' if taxonomy in {'background_semantic', 'scene_semantic'} else 'low_level_property' if taxonomy == 'local_detail' else 'object', 'target': evidence['target'], 'evidence_hint': evidence['evidence_hint'], 'hr_support': 'yes', 'language_prior_risk': 'medium' if taxonomy in {'object_semantic', 'background_semantic', 'scene_semantic'} else 'low', 'too_tiny_risk': 'medium' if taxonomy == 'text_logo' else 'low', 'low_detail_risk': 'medium'})
    return rows

def _low_patch_core(core: ModuleType, profile_name: str, qa_prompt_style: str) -> None:
    spec = _low_profile_spec(profile_name, core)
    taxonomies = spec['taxonomies']
    caps = _low_profile_caps(profile_name, core)
    core.TAXONOMIES = list(taxonomies)
    core.CAPS = caps
    core.EXPERIMENT_ID = f"realsr_4x_{spec.get('family', 'low')}_level_qa"
    core.build_inventory_prompt = lambda image_id: _low_evidence_prompt(_low_prompt_image_id(image_id), profile_name, core)
    core.build_qa_prompt = lambda image_id, taxonomy, rows: _low_qa_prompt(_low_prompt_image_id(image_id), taxonomy, rows, qa_prompt_style, profile_name, core)
    if _low_is_high_profile(profile_name):
        if profile_name == 'high-semantic-scene-upto144':
            core.normalize_inventory = lambda raw_items, image_id: _low_normalize_schema_aligned_high_inventory(raw_items, image_id, taxonomies, caps)
        else:
            core.normalize_inventory = lambda raw_items, image_id: _low_normalize_high_inventory(raw_items, image_id, taxonomies, caps)
        core.normalize_qas = _low_normalize_high_qas
    else:
        core.normalize_inventory = lambda raw_items, image_id: _low_normalize_inventory(raw_items, image_id, taxonomies, caps)
        core.normalize_qas = _low_normalize_qas

def model_core():
    return SimpleNamespace(**{'TAXONOMIES': _high_TAXONOMIES, 'DEFINITIONS': _high_DEFINITIONS, 'CAPS': _high_CAPS, 'SCENE_HIGH_TAXONOMIES': _high_SCENE_HIGH_TAXONOMIES, 'SCENE_HIGH_DEFINITIONS': _high_SCENE_HIGH_DEFINITIONS, 'HIGH_PROMPT_PROFILES': _high_HIGH_PROMPT_PROFILES, 'CURRENT_HIGH_EVIDENCE_PROMPT': _high_CURRENT_HIGH_EVIDENCE_PROMPT, 'CURRENT_HIGH_QA_PROMPT': _high_CURRENT_HIGH_QA_PROMPT, 'high_profile_spec': _high_high_profile_spec, 'build_high_inventory_prompt': _high_build_high_inventory_prompt, 'build_high_qa_prompt': _high_build_high_qa_prompt, 'load_model_and_processor': _high_load_model_and_processor, 'apply_generation_template': _high_apply_generation_template, 'generation_kwargs': _high_generation_kwargs, 'seed_generation': _high_seed_generation, 'build_messages': _high_build_messages, 'generate_text': _high_generate_text, 'generate_text_batch': _high_generate_text_batch, 'mean_answer_log_prob': _high_mean_answer_log_prob})

import json
import re
from scoring import PROTOCOL


def normalize_generation(raw, record, normalize, image_id):
    """Only invalid generated content becomes empty; execution errors still propagate."""
    error = None
    rows = []
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        items = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        items = None
    if items is None:
        error = "invalid_json_array"
    else:
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            error = "invalid_item_schema"
        else:
            rows = normalize(items)
            if items and not rows:
                error = "no_valid_items_after_normalization"
    record["status"] = "empty_fallback" if error else "complete"
    if error:
        record["error"] = error
        print(f"REVERSE_GENERATION_EMPTY image={image_id} stage={record['stage']} "
              f"taxonomy={record.get('taxonomy', 'all')} reason={error}", flush=True)
    return rows



def generate_questions(core, model, processor, image, image_id, device, generation_batch_size=1):
    """Use canonical prompt builders and normalizers, without rewriting prompts."""
    records, questions = [], []
    prompt = core.build_inventory_prompt(image_id)
    raw = core.generate_text(model, processor, image, prompt, device, PROTOCOL["max_new_tokens"])
    record = {"stage": "inventory", "prompt": prompt, "raw_generation": raw}
    records.append(record)
    inventory = normalize_generation(raw, record,
        lambda items: core.normalize_inventory(items, image_id), image_id)
    jobs = []
    for taxonomy in core.TAXONOMIES:
        evidence = [item for item in inventory if item["taxonomy"] == taxonomy]
        if not evidence:
            continue
        prompt = core.build_qa_prompt(image_id, taxonomy, evidence)
        jobs.append((taxonomy, evidence, prompt))
    raw_outputs = []
    if generation_batch_size < 1:
        raise ValueError("Generation batch size must be positive")
    for start in range(0, len(jobs), generation_batch_size):
        chunk = jobs[start:start + generation_batch_size]
        if generation_batch_size == 1:
            raw_outputs.append(core.generate_text(model, processor, image, chunk[0][2], device,
                                                  PROTOCOL["max_new_tokens"]))
        else:
            raw_outputs.extend(core.generate_text_batch(model, processor, [image] * len(chunk),
                [job[2] for job in chunk], device, PROTOCOL["max_new_tokens"]))
    for (taxonomy, evidence, prompt), raw in zip(jobs, raw_outputs):
        record = {"stage": "qa", "taxonomy": taxonomy, "prompt": prompt, "raw_generation": raw}
        records.append(record)
        normalized = normalize_generation(raw, record,
            lambda items: core.normalize_qas(items, image_id, taxonomy, evidence), image_id)
        questions.extend(normalized)
    questions.sort(key=lambda row: row["evidence_id"])
    questions = [{**q, "qa_id": f"{image_id}_qa{index:02d}"} for index, q in enumerate(questions)]
    errors = [{key: record[key] for key in ("stage", "taxonomy", "error") if key in record}
              for record in records if record["status"] == "empty_fallback"]
    return {"inventory": inventory, "questions": questions, "generations": records,
            "generation_status": "complete_with_empty_fallback" if errors else "complete",
            "generation_errors": errors}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, help='Optional folder for your own images')
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK_ROOT)
    parser.add_argument('--split', choices=('gt', 'sr'), default='gt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--exclude-folder', action='append', default=[])
    parser.add_argument('--generation-batch-size', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.images is None:
        pairs = BenchmarkDataset(args.benchmark)
        field = 'gt_path' if args.split == 'gt' else 'candidate_path'
        paths = sorted({Path(row[field]) for row in pairs})
    else:
        iterator = args.images.rglob('*') if args.recursive else args.images.iterdir()
        paths = sorted(p.resolve() for p in iterator if p.is_file()
                       and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
                       and not set(p.relative_to(args.images).parts[:-1]) & set(args.exclude_folder))
    args.output = args.output or OUTPUT_ROOT / (args.split + '_qa.jsonl')
    if not paths:
        raise ValueError('No input images')
    if args.generation_batch_size < 1:
        raise ValueError('Generation batch size must be positive')
    if args.dry_run:
        print(json.dumps({'images': len(paths), 'families': ['high', 'low'],
                          'model': args.model, 'output': str(args.output)}, indent=2))
        return
    from PIL import Image
    import transformers
    core = model_core()
    from scoring import PROFILES, PROTOCOL
    if transformers.__version__ != '4.57.6':
        raise RuntimeError('Use requirements.txt: Transformers 4.57.6 is required')
    os.umask(0)
    os.environ['QA_GENERATION_DO_SAMPLE'] = '0'
    core.seed_generation()
    done = {}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('Output exists; use --resume or a new output path')
        for line in args.output.read_text().splitlines():
            record = json.loads(line)
            key = (record['image_path'], record['family'])
            if key in done:
                raise ValueError(f'Duplicate completed record: {key}')
            done[key] = record
    model, processor = core.load_model_and_processor(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a') as stream:
        for path in paths:
            digest = sha256(path)
            with Image.open(path) as image:
                rgb = image.convert('RGB')
            for family, profile in PROFILES.items():
                identity = {'image_sha256': digest, 'protocol': PROTOCOL, 'model': args.model,
                            'generation_batch_size': args.generation_batch_size,
                            'prompt_sources': {p.name: sha256(p) for p in sorted(PROMPT_DIR.glob('*.txt'))},
                            'generation_source': sha256(Path(__file__))}
                previous = done.get((str(path), family))
                if previous:
                    if previous['identity'] != identity:
                        raise ValueError(f'Resume inputs changed: {path}/{family}')
                    continue
                _low_patch_core(core, profile, 'fixed')
                record = {'image_id': path.stem, 'image_path': str(path), 'family': family,
                          'identity': identity, **generate_questions(core, model, processor,
                              rgb, path.stem, args.device, args.generation_batch_size)}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                print(f"{path.name}/{family}: {len(record['questions'])} questions", flush=True)


if __name__ == '__main__':
    main()
