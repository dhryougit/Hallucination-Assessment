"""Answer-token support, bidirectional VHA scores, and pair inference."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path

from typing import Any

def expand_image_tokens(processor: Any, texts: list[str], image_grid_thw: Any) -> list[str]:
    merge_length = processor.image_processor.merge_size ** 2
    output = texts.copy()
    index = 0
    for text_index in range(len(output)):
        while processor.image_token in output[text_index]:
            num_image_tokens = int(image_grid_thw[index].prod().item() // merge_length)
            output[text_index] = output[text_index].replace(processor.image_token, '<|placeholder|>' * num_image_tokens, 1)
            index += 1
        output[text_index] = output[text_index].replace('<|placeholder|>', processor.image_token)
    return output

def cached_batch_mean_answer_log_prob(model: Any, processor: Any, gq: Any, image_inputs: dict[str, Any], visual_cache: tuple[Any, Any], question: str, answer: str, batch_count: int, device: str) -> list[float]:
    import torch
    prefixes = [processor.apply_chat_template(gq.build_messages(question, image='cached', for_scoring=True), tokenize=False, add_generation_prompt=True) for _ in range(batch_count)]
    prefixes = expand_image_tokens(processor, prefixes, image_inputs['image_grid_thw'])
    answer_counts = []
    for prefix in prefixes:
        prefix_ids = processor.tokenizer(prefix, add_special_tokens=False).input_ids
        full_ids = processor.tokenizer(prefix + answer, add_special_tokens=False).input_ids
        answer_counts.append(len(full_ids) - len(prefix_ids))
    text_inputs = processor.tokenizer([prefix + answer for prefix in prefixes], padding=True, return_tensors='pt').to(device)
    full_enc = {**text_inputs, **image_inputs}
    logits_to_keep = max(answer_counts) + 1
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
        input_ids = full_enc['input_ids']
        attention_mask = full_enc['attention_mask']
        image_grid_thw = full_enc['image_grid_thw']
        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        image_embeds, deepstack_image_embeds = visual_cache
        image_mask, _ = model.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))
        visual_pos_masks = image_mask[..., 0]
        position_ids, _rope_deltas = model.model.get_rope_index(input_ids, image_grid_thw, None, attention_mask=attention_mask)
        outputs = model.model.language_model(input_ids=None, position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_image_embeds, use_cache=False)
        logits = model.lm_head(outputs.last_hidden_state[:, -logits_to_keep:, :])
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    output: list[float] = []
    for index in range(batch_count):
        active = torch.nonzero(full_enc['attention_mask'][index], as_tuple=False).squeeze(1)
        full_end = int(active[-1].item()) + 1
        answer_start = full_end - answer_counts[index]
        answer_ids = full_enc['input_ids'][index, answer_start:full_end]
        if answer_ids.numel() == 0:
            output.append(-1000000000.0)
            continue
        answer_log_probs = log_probs[index, logits_to_keep - answer_counts[index] - 1:logits_to_keep - 1]
        token_log_probs = answer_log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
        output.append(float(token_log_probs.mean().item()))
    return output

import os

SCORE_BATCH_SIZE = int(os.environ.get('VQA_SCORE_BATCH_SIZE', '8'))

GENERATION_BATCH_SIZE = int(os.environ.get('VQA_GENERATION_BATCH_SIZE', '3'))

def score_requests(model, processor, core, scorer, questions, device, image_inputs=None, visual_cache=None, image_indices=None, batch_size=None):
    """Return one mean answer-token log probability per request, in input order."""
    import torch
    size = SCORE_BATCH_SIZE if batch_size is None else batch_size
    if size < 1:
        raise ValueError('VQA score batch size must be positive')
    if image_indices is not None and len(image_indices) != len(questions):
        raise ValueError('Question/image index count mismatch')
    result = []
    for start in range(0, len(questions), size):
        chunk = questions[start:start + size]
        indices = None if image_indices is None else image_indices[start:start + size]
        retry = False
        try:
            result.extend(_score_batch(model, processor, core, scorer, chunk, device, image_inputs, visual_cache, indices))
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            retry = True
        if retry:
            torch.cuda.empty_cache()
            print(f'VQA_SCORE_OOM_SPLIT batch={len(chunk)}', flush=True)
            result.extend(score_requests(model, processor, core, scorer, chunk, device, image_inputs, visual_cache, indices, max(1, len(chunk) // 2)))
    return result

def _score_batch(model, processor, core, scorer, questions, device, image_inputs, visual_cache, image_indices):
    import torch
    with_image = image_indices is not None
    prefixes = [processor.apply_chat_template(core.build_messages(q['question'], image='cached' if with_image else None, for_scoring=True), tokenize=False, add_generation_prompt=True) for q in questions]
    grid = embeds = deep = None
    if with_image:
        grid = image_inputs['image_grid_thw'][image_indices]
        prefixes = scorer.expand_image_tokens(processor, prefixes, grid)
        lengths = (image_inputs['image_grid_thw'].prod(dim=1) // processor.image_processor.merge_size ** 2).tolist()
        chunks = visual_cache[0].split(lengths)
        embeds = torch.cat([chunks[i] for i in image_indices])
        deep = []
        for layer in visual_cache[1]:
            chunks = layer.split(lengths)
            deep.append(torch.cat([chunks[i] for i in image_indices]))
    counts = [len(processor.tokenizer(prefix + q['answer'], add_special_tokens=False).input_ids) - len(processor.tokenizer(prefix, add_special_tokens=False).input_ids) for prefix, q in zip(prefixes, questions)]
    if any((n < 1 for n in counts)):
        raise ValueError('Empty answer token span')
    padding = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = 'left'
    try:
        text = processor.tokenizer([p + q['answer'] for p, q in zip(prefixes, questions)], padding=True, return_tensors='pt').to(device)
    finally:
        processor.tokenizer.padding_side = padding
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
        ids, mask = (text['input_ids'], text['attention_mask'])
        inputs_embeds = model.model.get_input_embeddings()(ids)
        visual_mask = None
        if with_image:
            placeholder, _ = model.model.get_placeholder_mask(ids, inputs_embeds=inputs_embeds, image_features=embeds)
            inputs_embeds = inputs_embeds.masked_scatter(placeholder, embeds.to(inputs_embeds.device, inputs_embeds.dtype))
            visual_mask = placeholder[..., 0]
        positions, _ = model.model.get_rope_index(ids, grid, None, attention_mask=mask)
        output = model.model.language_model(input_ids=None, position_ids=positions, attention_mask=mask, inputs_embeds=inputs_embeds, visual_pos_masks=visual_mask, deepstack_visual_embeds=deep, use_cache=False)
        hidden = torch.cat([output.last_hidden_state[i, -n - 1:-1] for i, n in enumerate(counts)])
        logits = model.lm_head(hidden)
    targets = torch.cat([ids[i, -n:] for i, n in enumerate(counts)])
    token_logp = torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
    return [float(values.mean()) for values in token_logp.split(counts)]

import math

from collections import defaultdict

FORWARD_SUPPORT_THRESHOLD = -2.0

PROFILES = {'high': 'high-semantic-scene-upto144', 'low': 'low-level-prompt'}

PROTOCOL = {'version': 3, 'profiles': PROFILES, 'qa_prompt_style': 'fixed', 'max_new_tokens': 1400, 'do_sample': False, 'support_threshold': 0.0, 'margin_threshold': 0.5, 'reverse_filter': 'none', 'forward_weight': 1.0, 'forward_support_threshold': FORWARD_SUPPORT_THRESHOLD, 'forward_filter': 'none', 'forward_margin_gate': 'none', 'denominator': 'all_questions', 'reverse_weight': 1.0, 'family_reduction': 'pool_questions', 'dataset_reduction': 'mean_image_scores', 'empty_reverse': 0.0, 'invalid_generation': 'empty_failed_stage_with_audit'}

def aggregate_forward(rows, expected, image_ids=None):
    """Image means first; High+Low pools questions, not family means."""
    expected_keys = {(q['family'], q['image_id'], q['qa_id']) for q in expected}
    keys = [(q['family'], q['image_id'], q['qa_id']) for q in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError('VQA coverage mismatch; partial validation is not allowed')
    by_image = defaultdict(list)
    for row in rows:
        if not all((math.isfinite(float(row[k])) for k in ('forward_drop', 'score_gt'))):
            raise ValueError('Non-finite VQA forward score')
        by_image[row['image_id']].append(row)
    result = {}
    for image_id in image_ids if image_ids is not None else by_image:
        qas = by_image[image_id]
        scores = {}
        for family in ('high', 'low', 'high_low'):
            subset = qas if family == 'high_low' else [q for q in qas if q['family'] == family]
            scores[f'vqa_forward_{family}'] = sum((float(q['forward_drop']) for q in subset if float(q['score_gt']) > FORWARD_SUPPORT_THRESHOLD)) / len(subset) if subset else None
            scores[f'vqa_count_{family}'] = len(subset)
        result[image_id] = scores
    return result

def reverse_term(score_sr, score_gt):
    if not all((math.isfinite(value) for value in (score_sr, score_gt))):
        raise ValueError('Non-finite reverse support')
    margin = score_sr - score_gt
    return max(margin - PROTOCOL['margin_threshold'], 0.0) if score_sr > PROTOCOL['support_threshold'] else 0.0

def aggregate_bidirectional(forward, records, image_ids):
    """Keep zero-contribution questions in the denominator; pool High+Low Q/A."""
    expected = {(image_id, family) for image_id in image_ids for family in PROFILES}
    keys = [(r['image_id'], r['family']) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Reverse image/family coverage mismatch (empty Q/A needs a completed record)')
    by_image = {image_id: [] for image_id in image_ids}
    for record in records:
        if record['status'] != 'complete':
            raise ValueError('Incomplete reverse record')
        questions = record['rows']
        if len({r['qa_id'] for r in questions}) != len(questions):
            raise ValueError('Duplicate reverse Q/A')
        by_image[record['image_id']].extend(({**q, 'family': record['family']} for q in questions))
    result = {}
    for image_id, questions in by_image.items():
        values = dict(forward[image_id])
        for family in (*PROFILES, 'high_low'):
            subset = questions if family == 'high_low' else [q for q in questions if q['family'] == family]
            terms = [reverse_term(float(q['score_sr']), float(q['score_gt'])) for q in subset]
            reverse = sum(terms) / len(terms) if terms else 0.0
            f = values[f'vqa_forward_{family}']
            values.update({f'vqa_reverse_{family}': reverse, f'vqa_final_{family}': f + reverse if f is not None else None, f'reverse_qa_count_{family}': len(subset), f'reverse_active_count_{family}': sum((t > 0 for t in terms))})
        result[image_id] = values
    return result

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluation_summary(rows):
    from scipy.stats import pearsonr, spearmanr
    if not rows or not all(row.get('human_score') not in (None, '') for row in rows):
        return None
    target = [float(row['human_score']) for row in rows]
    prediction = [float(row['vha']) for row in rows]
    result = {'pairs': len(rows)}
    for name, function in [('pooled_spearman', spearmanr), ('pooled_pearson', pearsonr)]:
        if len(rows) < 2 or len(set(target)) < 2 or len(set(prediction)) < 2:
            result[name] = None
        else:
            value = float(function(target, prediction).statistic)
            result[name] = value if math.isfinite(value) else None
    return result


def score_pairs(pairs, gt_qa, sr_qa, args):
    for pair in pairs:
        for field, records in [('gt_path', gt_qa), ('candidate_path', sr_qa)]:
            digest = sha256(pair[field])
            for family in PROFILES:
                record = records[(pair[field], family)]
                if record['identity']['image_sha256'] != digest:
                    raise ValueError(f'Image changed since Q/A generation: {pair[field]}')
                if record['identity']['protocol'] != PROTOCOL or record['identity']['model'] != args.model:
                    raise ValueError('Q/A protocol/model mismatch')
    if args.dry_run:
        print(json.dumps({'pairs': len(pairs), 'protocol': PROTOCOL}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve previous scores')
    import torch
    import transformers
    from PIL import Image
    from qa_generation import model_core
    core = model_core()
    scorer = sys.modules[__name__]
    if transformers.__version__ != '4.57.6':
        raise RuntimeError('Use requirements.txt: Transformers 4.57.6 is required')
    model, processor = core.load_model_and_processor(args.model, args.device)
    args.output.mkdir(parents=True)
    predictions = []
    with (args.output / 'qa_scores.jsonl').open('w') as detail:
        for pair in pairs:
            with Image.open(pair['gt_path']) as gt, Image.open(pair['candidate_path']) as sr:
                inputs = processor.image_processor(images=[gt.convert('RGB'), sr.convert('RGB')],
                                                   return_tensors='pt')
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                embeds, deep = model.model.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'])
                visual = (torch.cat(embeds, dim=0), deep)
            forward, reverse = [], []
            for family in PROFILES:
                for direction, records in [('forward', gt_qa), ('reverse', sr_qa)]:
                    field = 'gt_path' if direction == 'forward' else 'candidate_path'
                    questions = records[(pair[field], family)]['questions']
                    priors = score_requests(model, processor, core, scorer, questions, args.device)
                    paired = [q for q in questions for _ in range(2)]
                    values = score_requests(model, processor, core, scorer, paired, args.device,
                                           inputs, visual, [0, 1] * len(questions))
                    rows = []
                    for index, q in enumerate(questions):
                        gt_logp, sr_logp = values[2 * index:2 * index + 2]
                        row = {**q, 'image_id': pair['sample_id'], 'family': family,
                               'text_prior': priors[index], 'gt_logprob_with_image': gt_logp,
                               'sr_logprob_with_image': sr_logp,
                               'score_gt': gt_logp - priors[index],
                               'score_sr': sr_logp - priors[index],
                               'forward_drop': gt_logp - sr_logp}
                        rows.append(row)
                        detail.write(json.dumps({'sample_id': pair['sample_id'],
                                                 'direction': direction, **row}, allow_nan=False) + '\n')
                    if direction == 'forward':
                        forward.extend(rows)
                    else:
                        reverse.append({'image_id': pair['sample_id'], 'family': family,
                                        'status': 'complete', 'rows': rows})
            aggregate = aggregate_forward(forward, forward, [pair['sample_id']])
            result = aggregate_bidirectional(aggregate, reverse, [pair['sample_id']])[pair['sample_id']]
            if result['vqa_final_high_low'] is None:
                raise ValueError(f"No GT questions for {pair['sample_id']}; VHA is undefined")
            predictions.append({**pair, 'vha': result['vqa_final_high_low'], **result})
            write_csv(args.output / 'pair_scores.csv', predictions)
            detail.flush()
            print(pair['sample_id'], result['vqa_final_high_low'], flush=True)
            del inputs, visual, embeds, deep
    summary = evaluation_summary(predictions)
    if summary is not None:
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(json.dumps(summary, indent=2))
    (args.output / 'protocol.json').write_text(json.dumps(PROTOCOL, indent=2) + '\n')


def score_image_questions(model, processor, core, questions, images, device):
    """Raw answer support for one or more images; shared by training-data construction."""
    if not questions:
        return []
    import torch
    scorer = sys.modules[__name__]
    inputs = processor.image_processor(images=images, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                         enabled=str(device).startswith('cuda')):
        embeds, deep = model.model.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'])
        visual = (torch.cat(embeds, dim=0), deep)
    priors = score_requests(model, processor, core, scorer, questions, device)
    count = len(images)
    repeated = [q for q in questions for _ in images]
    values = score_requests(model, processor, core, scorer, repeated, device,
                            inputs, visual, list(range(count)) * len(questions))
    return [{**q, 'text_prior': priors[index],
             'image_logprobs': values[index * count:(index + 1) * count],
             'scores': [v - priors[index] for v in values[index * count:(index + 1) * count]]}
            for index, q in enumerate(questions)]
