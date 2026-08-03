"""Shared, model-neutral support for VStar pre-commit routing evaluations."""

from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from constants import ALL_IMG_TOKENS_STR, COT_ACTIVATION, DEFAULT_GRD_TOKEN
from model.load_model import _build_inference_batch


ORACLE_BOX_COORDINATE_SYSTEM = (
    'normalized_xyxy_on_center_padded_square'
)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path, records):
    """Atomically replace one JSONL file with the supplied records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=str(path.parent),
        prefix='.' + path.name + '.',
        suffix='.tmp',
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for record in records:
                handle.write(json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def latest_records_by_question_id(records):
    """Keep the final row for each sample while preserving first-seen order."""
    order = []
    latest = {}
    for record in records:
        question_id = record.get('question_id')
        if not isinstance(question_id, str) or not question_id:
            raise ValueError('result record lacks a non-empty question_id')
        if question_id not in latest:
            order.append(question_id)
        latest[question_id] = record
    return [latest[question_id] for question_id in order]


def record_events(records):
    return [
        event
        for record in records
        if record.get('status') == 'ok'
        for event in (record.get('intervention') or {}).get('events', [])
    ]


def append_events(path, events):
    if not events:
        return
    with Path(path).open('a', encoding='utf-8') as handle:
        for event in events:
            handle.write(json.dumps(
                event,
                ensure_ascii=False,
                allow_nan=False,
            ) + '\n')
        handle.flush()


def make_conversation(question):
    return [{
        'from': 'human',
        'value': (
            ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n'
            + question + ' ' + COT_ACTIVATION
        ),
    }]


def score_options(
        model, preprocessor, image, conversation, options, generated_ids,
        max_new_tokens, temperature, likelihood_reduction):
    """Reuse VoCoT's option-conditional likelihood scoring with a fixed CoT."""
    batch = _build_inference_batch(
        preprocessor,
        image,
        conversation=conversation,
        options=options,
    )
    prompt = batch['input_ids']
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    thought_ids = torch.cat([
        prompt.to(model.device),
        torch.tensor(
            generated_ids,
            dtype=prompt.dtype,
            device=model.device,
        ).unsqueeze(0),
    ], dim=1)
    with torch.inference_mode():
        prediction, _ = model.calculate_options(
            batch,
            cot=True,
            further_instruct=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            likelihood_reduction=likelihood_reduction,
            thought_override_ids=thought_ids,
        )
    return int(prediction)


def transition_counts(records):
    counts = Counter({
        'correct_to_wrong': 0,
        'wrong_to_correct': 0,
        'correct_to_correct': 0,
        'wrong_to_wrong': 0,
    })
    for record in records:
        before = bool(record['baseline_prediction_correct'])
        after = bool(record['router_prediction_correct'])
        if before and not after:
            counts['correct_to_wrong'] += 1
        elif not before and after:
            counts['wrong_to_correct'] += 1
        elif before:
            counts['correct_to_correct'] += 1
        else:
            counts['wrong_to_wrong'] += 1
    return dict(counts)


def exact_mcnemar_pvalue(wrong_to_correct, correct_to_wrong):
    discordant = int(wrong_to_correct) + int(correct_to_wrong)
    if discordant == 0:
        return 1.0
    tail = min(int(wrong_to_correct), int(correct_to_wrong))
    probability = sum(
        math.comb(discordant, value)
        for value in range(tail + 1)
    )
    return min(1.0, 2.0 * probability / float(2 ** discordant))


def paired_metrics(records):
    if not records:
        return {'samples': 0}
    transitions = transition_counts(records)
    baseline_correct = sum(
        bool(record['baseline_prediction_correct']) for record in records
    )
    router_correct = sum(
        bool(record['router_prediction_correct']) for record in records
    )
    return {
        'samples': len(records),
        'baseline_correct_count': baseline_correct,
        'baseline_accuracy': baseline_correct / len(records),
        'router_correct_count': router_correct,
        'router_accuracy': router_correct / len(records),
        'router_minus_baseline': (
            router_correct - baseline_correct
        ) / len(records),
        'answer_changed_count': sum(
            record['baseline_prediction'] != record['router_prediction']
            for record in records
        ),
        'correctness_transitions': transitions,
        'mcnemar_exact_two_sided_pvalue': exact_mcnemar_pvalue(
            transitions['wrong_to_correct'],
            transitions['correct_to_wrong'],
        ),
    }


__all__ = [
    'ORACLE_BOX_COORDINATE_SYSTEM',
    'append_events',
    'atomic_write_jsonl',
    'latest_records_by_question_id',
    'make_conversation',
    'paired_metrics',
    'read_jsonl',
    'record_events',
    'score_options',
]
