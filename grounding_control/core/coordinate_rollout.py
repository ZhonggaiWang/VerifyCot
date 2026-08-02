"""Policy-neutral mechanics for pre-commit VoCoT coordinate rollouts.

The shared base stops generation immediately after a candidate ``</coor>``
and before that candidate enters REFbind.  Policy-specific controllers decide
which coordinate to commit; the selected coordinate is injected exactly once
during the next clean replay.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN
from utils.coordinate_intervention import find_coordinate_spans

from ..contracts.boxes import Box, validate_normalized_box


class _ForcePrefixProcessor(LogitsProcessor):
    """Deterministically replay a committed generated-token prefix."""

    def __init__(self, prompt_length: int, prefix_ids: Sequence[int]):
        self.prompt_length = int(prompt_length)
        self.prefix_ids = list(prefix_ids)
        self.released = not self.prefix_ids

    @staticmethod
    def _force(scores: torch.FloatTensor, token_id: int) -> torch.FloatTensor:
        scores.fill_(float('-inf'))
        scores[:, int(token_id)] = 0
        return scores

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[0, self.prompt_length:].tolist()
        if len(generated) < len(self.prefix_ids):
            if generated != self.prefix_ids[:len(generated)]:
                raise RuntimeError('committed prefix diverged during clean replay')
            return self._force(scores, self.prefix_ids[len(generated)])
        if generated[:len(self.prefix_ids)] != self.prefix_ids:
            raise RuntimeError('clean replay did not preserve the committed prefix')
        self.released = True
        return scores


class _StopAfterNewCoordinate(StoppingCriteria):
    """Stop after a newly generated EOC and before its REFbind forward pass."""

    def __init__(self, minimum_sequence_length: int, eoc_token_id: int):
        self.minimum_sequence_length = int(minimum_sequence_length)
        self.eoc_token_id = int(eoc_token_id)
        self.triggered = False

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs) -> bool:
        self.triggered = bool(
            input_ids.shape[1] > self.minimum_sequence_length
            and input_ids[0, -1].item() == self.eoc_token_id
        )
        return self.triggered


@dataclass
class GenerationBoundary:
    generated_ids: List[int]
    candidate_span: Optional[Tuple[int, int]]
    candidate_box: Optional[Box]
    replayed_bound_box_count: int


@dataclass
class PrecommitInferenceResult:
    response: str
    generated_ids: List[int]
    status: str
    events: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            'response': self.response,
            'generated_ids': self.generated_ids,
            'status': self.status,
            'events': self.events,
        }


class CoordinateRolloutBase:
    """Shared generation/replay machinery with no verifier policy imports."""

    def __init__(
            self,
            model,
            tokenizer,
            batch_factory: Callable[[], Dict[str, Any]],
            sample_id: str,
            log_path: Optional[str] = None,
            sample_context: Optional[Mapping[str, Any]] = None):
        if not sample_id:
            raise ValueError('sample_id is required')
        self.model = model
        self.tokenizer = tokenizer
        self.batch_factory = batch_factory
        self.sample_id = str(sample_id)
        self.log_path = Path(log_path) if log_path else None
        self.sample_context = dict(sample_context or {})
        self.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
        self.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
        if self.boc_token_id is None or self.eoc_token_id is None:
            raise ValueError('tokenizer does not provide VoCoT coordinate tokens')

    @staticmethod
    def _validate_box(values: Sequence[float]) -> Box:
        return validate_normalized_box(values)

    def _box_from_span(
            self,
            generated_ids: Sequence[int],
            span: Tuple[int, int]) -> Box:
        text = self.tokenizer.decode(
            generated_ids[span[0]:span[1] + 1],
            skip_special_tokens=False,
        )
        match = re.search(
            re.escape(DEFAULT_BOC_TOKEN)
            + r'\s*([01](?:\.\d+)?)\s*,\s*([01](?:\.\d+)?)\s*,\s*'
            + r'([01](?:\.\d+)?)\s*,\s*([01](?:\.\d+)?)\s*'
            + re.escape(DEFAULT_EOC_TOKEN),
            text,
        )
        if match is None:
            raise ValueError(
                f'could not parse completed generated coordinate: {text!r}'
            )
        return self._validate_box(tuple(float(value) for value in match.groups()))

    def _object_reference(self, h_t_ids: Sequence[int]) -> str:
        """Extract local text after the previous completed coordinate."""

        last_eoc = -1
        for index, token_id in enumerate(h_t_ids):
            if token_id == self.eoc_token_id:
                last_eoc = index
        text = self.tokenizer.decode(
            h_t_ids[last_eoc + 1:],
            skip_special_tokens=False,
        )
        text = re.sub(r'\s+', ' ', text).strip()
        return text[-400:] or 'the current object reference'

    def _generate_until_next_coordinate(
            self,
            persistent_ids: Sequence[int],
            max_new_tokens: int,
            temperature: float) -> GenerationBoundary:
        """Replay committed history and stop at the next uncommitted EOC."""

        batch = self.batch_factory()
        prompt_length = int(batch['input_ids'].shape[-1])
        persistent = list(persistent_ids)
        processor = _ForcePrefixProcessor(prompt_length, persistent)
        stopper = _StopAfterNewCoordinate(
            minimum_sequence_length=prompt_length + len(persistent),
            eoc_token_id=self.eoc_token_id,
        )
        _, _, sequences = self.model.condition_completion(
            batch,
            avoid_image_gen=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            logits_processor=LogitsProcessorList([processor]),
            stopping_criteria=StoppingCriteriaList([stopper]),
            record_bound_boxes=True,
        )
        generated = sequences[0, prompt_length:].detach().cpu().tolist()
        if generated[:len(persistent)] != persistent:
            raise RuntimeError('generation diverged from committed history')

        spans = find_coordinate_spans(
            generated,
            self.boc_token_id,
            self.eoc_token_id,
        )
        candidate_span = None
        if stopper.triggered:
            candidates = [span for span in spans if span[0] >= len(persistent)]
            if not candidates:
                raise RuntimeError(
                    'stopped at </coor> without a new coordinate span'
                )
            candidate_span = candidates[-1]
        candidate_box = (
            None
            if candidate_span is None
            else self._box_from_span(generated, candidate_span)
        )

        expected_bound_count = len(find_coordinate_spans(
            persistent,
            self.boc_token_id,
            self.eoc_token_id,
        ))
        bound_box_count = len(getattr(self.model, 'last_bound_boxes', None) or [])
        if bound_box_count != expected_bound_count:
            raise RuntimeError(
                'candidate coordinate entered REFbind before routing: '
                f'expected {expected_bound_count} committed boxes, '
                f'observed {bound_box_count}'
            )
        return GenerationBoundary(
            generated_ids=generated,
            candidate_span=candidate_span,
            candidate_box=candidate_box,
            replayed_bound_box_count=bound_box_count,
        )

    def _encode_expert_coordinate(
            self,
            box: Sequence[float]) -> Tuple[List[int], Box]:
        """Encode and round-trip validate an expert box without a model pass."""

        normalized_box = self._validate_box(box)
        payload = ','.join(f'{value:.3f}' for value in normalized_box)
        suffix_ids = [self.boc_token_id] + self.tokenizer(
            payload + DEFAULT_EOC_TOKEN,
            add_special_tokens=False,
        ).input_ids
        if suffix_ids[-1] != self.eoc_token_id:
            raise RuntimeError('could not tokenize expert coordinate closing tag')
        spans = find_coordinate_spans(
            suffix_ids,
            self.boc_token_id,
            self.eoc_token_id,
        )
        if spans != [(0, len(suffix_ids) - 1)]:
            raise RuntimeError(
                'encoded expert result must contain exactly one complete '
                'coordinate span'
            )
        encoded_box = self._box_from_span(suffix_ids, spans[0])
        rounded_requested_box = tuple(
            float(f'{value:.3f}') for value in normalized_box
        )
        if any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(encoded_box, rounded_requested_box)):
            raise RuntimeError('encoded coordinate text differs from the expert box')
        return suffix_ids, encoded_box

    @staticmethod
    def _box_list(box: Optional[Box]):
        return None if box is None else [float(value) for value in box]

    def _write_event(self, event: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')


__all__ = [
    'CoordinateRolloutBase',
    'GenerationBoundary',
    'PrecommitInferenceResult',
]
