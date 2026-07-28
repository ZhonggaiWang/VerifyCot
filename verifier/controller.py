"""Clean-prefill pre-commit controller for VoCoT coordinates.

This first implementation intentionally does *not* reuse KV caches.  Each
generation call starts from the original multimodal prompt and force-replays
only persistent accepted tokens.  Consequently every accepted ``</coor>``
again executes VoCoT's native REFbind path, while rejected candidates and all
repair feedback are absent from the next clean prefill.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN
from utils.coordinate_intervention import find_coordinate_spans

from .prompts import (
    RepairMode,
    build_repair_prompt,
    build_repair_prompt_text_only_q,
)
from .stored_oracle import StoredOracleVerifier
from .types import Box, VerificationLookup


class _ForcePrefixProcessor(LogitsProcessor):
    """Deterministically replay a clean persistent/temporary prefix."""

    def __init__(self, prompt_length: int, prefix_ids: Sequence[int]):
        self.prompt_length = int(prompt_length)
        self.prefix_ids = list(prefix_ids)
        self.released = not self.prefix_ids

    @staticmethod
    def _force(scores: torch.FloatTensor, token_id: int) -> torch.FloatTensor:
        scores.fill_(float('-inf'))
        scores[:, token_id] = 0
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[0, self.prompt_length:].tolist()
        if len(generated) < len(self.prefix_ids):
            expected = self.prefix_ids[:len(generated)]
            if generated != expected:
                raise RuntimeError('clean prefill diverged while replaying persistent tokens')
            return self._force(scores, self.prefix_ids[len(generated)])
        if generated[:len(self.prefix_ids)] != self.prefix_ids:
            raise RuntimeError('clean prefill did not preserve the requested prefix')
        self.released = True
        return scores


class _StopAfterNewCoordinate(StoppingCriteria):
    """Stop immediately after an EOC generated beyond the forced prefix.

    Hugging Face evaluates stopping criteria immediately after appending the
    generated EOC, before the next forward pass.  That is what prevents
    ``prepare_inputs_for_generation`` from binding the candidate feature.
    """

    def __init__(self, minimum_sequence_length: int, eoc_token_id: int):
        self.minimum_sequence_length = int(minimum_sequence_length)
        self.eoc_token_id = int(eoc_token_id)
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.triggered = bool(
            input_ids.shape[1] > self.minimum_sequence_length
            and input_ids[0, -1].item() == self.eoc_token_id
        )
        return self.triggered


class _StopAtForcedCoordinateEnd(StoppingCriteria):
    """Stop exactly when a forced candidate's closing EOC is appended."""

    def __init__(self, target_sequence_length: int, eoc_token_id: int):
        self.target_sequence_length = int(target_sequence_length)
        self.eoc_token_id = int(eoc_token_id)
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.triggered = bool(
            input_ids.shape[1] == self.target_sequence_length
            and input_ids[0, -1].item() == self.eoc_token_id
        )
        return self.triggered


class _ForcePrefixSuppressNextBocProcessor(LogitsProcessor):
    """Replay H_t, then prohibit only its immediate next token from being BOC."""

    def __init__(self, prompt_length: int, prefix_ids: Sequence[int], boc_token_id: int):
        self.prefix = _ForcePrefixProcessor(prompt_length, prefix_ids)
        self.prompt_length = int(prompt_length)
        self.prefix_length = len(prefix_ids)
        self.boc_token_id = int(boc_token_id)
        self.suppressed_first_free_boc = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = self.prefix(input_ids, scores)
        generated_length = input_ids.shape[1] - self.prompt_length
        if generated_length >= self.prefix_length and not self.suppressed_first_free_boc:
            # This is exactly the first decoder choice after H_t.  All later
            # choices, including a possible later grounding decision, are free.
            scores[:, self.boc_token_id] = float('-inf')
            self.suppressed_first_free_boc = True
        return scores


@dataclass(frozen=True)
class RepairBaseSnapshot:
    """Textual H_t snapshot; KV cache is deliberately unused in version one."""

    repair_base_input_ids: Tuple[int, ...]
    repair_base_attention_mask: Tuple[int, ...]
    repair_base_position: int
    repair_base_kv_cache: None = None


@dataclass
class _GenerationBoundary:
    generated_ids: List[int]
    candidate_span: Optional[Tuple[int, int]]
    candidate_box: Optional[Box]
    bound_box_count: int


@dataclass
class VerifierInferenceResult:
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


class VerifierController:
    """Orchestrate candidate verification, repair, clean commit, and logging."""

    def __init__(self, model, tokenizer, batch_factory: Callable[[], Dict[str, Any]],
                 verifier: StoredOracleVerifier, sample_id: str,
                 repair_mode: RepairMode = 'typed_feedback',
                 accept_confidence: float = 0.8, max_retries: int = 2,
                 on_failure: str = 'skip_grounding_and_continue', log_path: Optional[str] = None):
        if not sample_id:
            raise ValueError('sample_id is required when verifier is enabled')
        if repair_mode not in {
            'blind_retry', 'binary_feedback', 'typed_feedback', 'concise_typed_feedback',
            'separated_reference_feedback', 'separated_reference_feedback_v2'
        }:
            raise ValueError(
                'repair_mode must be blind_retry, binary_feedback, typed_feedback, '
                'concise_typed_feedback, separated_reference_feedback, or '
                'separated_reference_feedback_v2'
            )
        if not 0.0 <= float(accept_confidence) <= 1.0:
            raise ValueError('accept_confidence must be in [0, 1]')
        if int(max_retries) < 0:
            raise ValueError('max_retries must be non-negative')
        if on_failure not in {'abort_sample', 'skip_grounding_and_continue'}:
            raise ValueError('on_failure must be abort_sample or skip_grounding_and_continue')
        self.model = model
        self.tokenizer = tokenizer
        self.batch_factory = batch_factory
        self.verifier = verifier
        self.sample_id = str(sample_id)
        self.repair_mode = repair_mode
        self.accept_confidence = float(accept_confidence)
        self.max_retries = int(max_retries)
        self.on_failure = on_failure
        self.log_path = Path(log_path) if log_path else None
        self.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
        self.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
        if self.boc_token_id is None or self.eoc_token_id is None:
            raise ValueError('tokenizer does not provide VoCoT coordinate tokens')

    @staticmethod
    def _validate_box(values: Sequence[float]) -> Box:
        if len(values) != 4:
            raise ValueError('generated coordinate must contain four values')
        box = tuple(float(value) for value in values)
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError(f'generated invalid normalized coordinate: {box}')
        return box  # type: ignore[return-value]

    def _box_from_span(self, generated_ids: Sequence[int], span: Tuple[int, int]) -> Box:
        text = self.tokenizer.decode(generated_ids[span[0]:span[1] + 1], skip_special_tokens=False)
        match = re.search(
            re.escape(DEFAULT_BOC_TOKEN)
            + r'\s*([01](?:\.\d+)?)\s*,\s*([01](?:\.\d+)?)\s*,\s*'
            + r'([01](?:\.\d+)?)\s*,\s*([01](?:\.\d+)?)\s*'
            + re.escape(DEFAULT_EOC_TOKEN),
            text,
        )
        if match is None:
            raise ValueError(f'could not parse completed generated coordinate: {text!r}')
        return self._validate_box(tuple(float(value) for value in match.groups()))

    def _object_reference(self, h_t_ids: Sequence[int]) -> str:
        """Conservative first-version reference extraction for prompts/logging.

        It deliberately takes only text after the previous completed coordinate,
        because earlier CoT mentions often refer to a different object.  It is
        not used as an oracle lookup key, so imperfect linguistic extraction
        cannot change an ACCEPT into an oracle decision.
        """
        last_eoc = -1
        for index, token_id in enumerate(h_t_ids):
            if token_id == self.eoc_token_id:
                last_eoc = index
        text = self.tokenizer.decode(h_t_ids[last_eoc + 1:], skip_special_tokens=False)
        text = re.sub(r'\s+', ' ', text).strip()
        # The segment is retained rather than attempting unsafe coreference
        # resolution.  A bounded value keeps JSONL logs practical.
        return text[-400:] or 'the current object reference'

    def _run_until_coordinate(self, persistent_ids: Sequence[int], temporary_ids: Sequence[int],
                              max_new_tokens: int, temperature: float,
                              suppress_refbind_generated_eoc_indices: Optional[Sequence[int]] = None
                              ) -> _GenerationBoundary:
        """Freshly prefill accepted history, then stop after one candidate EOC."""
        batch = self.batch_factory()
        prompt_length = int(batch['input_ids'].shape[-1])
        forced_prefix = list(persistent_ids) + list(temporary_ids)
        processor = _ForcePrefixProcessor(prompt_length, forced_prefix)
        stopper = _StopAfterNewCoordinate(
            minimum_sequence_length=prompt_length + len(forced_prefix),
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
            suppress_refbind_generated_eoc_indices=suppress_refbind_generated_eoc_indices,
        )

        generated = sequences[0, prompt_length:].detach().cpu().tolist()
        spans = find_coordinate_spans(generated, self.boc_token_id, self.eoc_token_id)
        candidate_span = None
        if stopper.triggered:
            candidates = [span for span in spans if span[0] >= len(persistent_ids)]
            if not candidates:
                raise RuntimeError('stopped at </coor> but no new coordinate span was found')
            candidate_span = candidates[-1]
        candidate_box = None if candidate_span is None else self._box_from_span(generated, candidate_span)
        bound_box_count = len(getattr(self.model, 'last_bound_boxes', None) or [])
        # In the simplified visual-aware repair setting, a rejected coordinate
        # written into temporary feedback follows Volcano's normal REFbind path
        # inside the sandbox. The candidate EOC still stops before its feature
        # can be injected.
        completed_prefix_coordinate_count = len(find_coordinate_spans(
            forced_prefix, self.boc_token_id, self.eoc_token_id
        ))
        suppressed_prefix_coordinate_count = sum(
            1 for index in (suppress_refbind_generated_eoc_indices or [])
            if 1 <= int(index) <= completed_prefix_coordinate_count
        )
        expected_bound_count = completed_prefix_coordinate_count - suppressed_prefix_coordinate_count
        if bound_box_count != expected_bound_count:
            raise RuntimeError(
                'unexpected REFbind injection while candidate/feedback was uncommitted: '
                f'expected {expected_bound_count}, observed {bound_box_count}'
            )
        return _GenerationBoundary(generated, candidate_span, candidate_box, bound_box_count)

    def _continue_after_failed_grounding(self, h_t_ids: Sequence[int], max_new_tokens: int,
                                         temperature: float) -> Tuple[List[int], bool]:
        """Skip the current object's immediate BOC and freely finish the rollout."""
        batch = self.batch_factory()
        prompt_length = int(batch['input_ids'].shape[-1])
        processor = _ForcePrefixSuppressNextBocProcessor(
            prompt_length, h_t_ids, self.boc_token_id
        )
        _, _, sequences = self.model.condition_completion(
            batch,
            avoid_image_gen=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            logits_processor=LogitsProcessorList([processor]),
            record_bound_boxes=True,
        )
        generated = sequences[0, prompt_length:].detach().cpu().tolist()
        if not processor.suppressed_first_free_boc:
            raise RuntimeError('failed to reach the first free decoding step after H_t')
        return generated, processor.suppressed_first_free_boc

    def _run_forced_coordinate(self, prefix_ids: Sequence[int], box: Sequence[float],
                               max_new_tokens: int, temperature: float) -> _GenerationBoundary:
        """Replay a reference prefix and stop at a deliberately forced box.

        The forced EOC is the final generated token, so its feature is not
        injected yet.  This is the pre-commit candidate presented to the
        stored verifier.
        """
        payload = ','.join(f'{float(value):.3f}' for value in box) + DEFAULT_EOC_TOKEN
        suffix_ids = [self.boc_token_id] + self.tokenizer(
            payload, add_special_tokens=False
        ).input_ids
        if suffix_ids[-1] != self.eoc_token_id:
            raise RuntimeError('could not tokenize forced coordinate closing tag')
        batch = self.batch_factory()
        prompt_length = int(batch['input_ids'].shape[-1])
        forced_prefix = list(prefix_ids) + suffix_ids
        processor = _ForcePrefixProcessor(prompt_length, forced_prefix)
        stopper = _StopAtForcedCoordinateEnd(
            target_sequence_length=prompt_length + len(forced_prefix),
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
        if not stopper.triggered or generated != forced_prefix:
            raise RuntimeError('failed to replay reference prefix and force the intervention box')
        spans = find_coordinate_spans(generated, self.boc_token_id, self.eoc_token_id)
        candidates = [span for span in spans if span[0] >= len(prefix_ids)]
        if not candidates:
            raise RuntimeError('forced intervention did not produce a coordinate span')
        candidate_span = candidates[-1]
        candidate_box = self._box_from_span(generated, candidate_span)
        expected_bound_count = len(find_coordinate_spans(
            prefix_ids, self.boc_token_id, self.eoc_token_id
        ))
        bound_box_count = len(getattr(self.model, 'last_bound_boxes', None) or [])
        if bound_box_count != expected_bound_count:
            raise RuntimeError('forced candidate unexpectedly entered REFbind before verification')
        return _GenerationBoundary(generated, candidate_span, candidate_box, bound_box_count)

    def _continue_freely(self, persistent_ids: Sequence[int], max_new_tokens: int,
                         temperature: float) -> List[int]:
        """Cleanly replay accepted state, then free-run without any verifier."""
        batch = self.batch_factory()
        prompt_length = int(batch['input_ids'].shape[-1])
        processor = _ForcePrefixProcessor(prompt_length, persistent_ids)
        _, _, sequences = self.model.condition_completion(
            batch,
            avoid_image_gen=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            logits_processor=LogitsProcessorList([processor]),
            record_bound_boxes=True,
        )
        generated = sequences[0, prompt_length:].detach().cpu().tolist()
        if generated[:len(persistent_ids)] != list(persistent_ids):
            raise RuntimeError('free continuation diverged while replaying committed state')
        return generated

    def run_one_shot_reference_repair(self, reference_generated_ids: Sequence[int],
                                      selected_coordinate_index: int, random_box: Sequence[float],
                                      max_new_tokens: int = 1024,
                                      temperature: float = 0.0,
                                      suppress_refbind_for_random_box: bool = False
                                      ) -> VerifierInferenceResult:
        """Repair one forced random box on a saved oracle reference trajectory.

        Only the selected coordinate is verified.  Its replacement is committed
        once without a second verifier call; all later CoT decoding is free.
        ``suppress_refbind_for_random_box`` keeps literal q text but skips only
        q's temporary visual binding in the repair sandbox.
        """
        spans = find_coordinate_spans(reference_generated_ids, self.boc_token_id, self.eoc_token_id)
        selected_offset = int(selected_coordinate_index) - 1
        if not 0 <= selected_offset < len(spans):
            raise ValueError(f'selected_coordinate_index must be in [1, {len(spans)}]')
        target_span = spans[selected_offset]
        h_t_ids = list(reference_generated_ids[:target_span[0]])
        forced = self._run_forced_coordinate(h_t_ids, random_box, max_new_tokens, temperature)
        assert forced.candidate_box is not None and forced.candidate_span is not None
        lookup = self.verifier.verify(
            self.sample_id, int(selected_coordinate_index), 0, forced.candidate_box
        )
        result = lookup.result
        if (lookup.missing_oracle_record or lookup.oracle_candidate_mismatch
                or result.verdict != 'misaligned' or result.reason != 'wrong_object'):
            raise RuntimeError(
                'one-shot random intervention requires a matching stored '
                'misaligned/wrong_object oracle record'
            )

        object_reference = self._object_reference(h_t_ids)
        repair_text = build_repair_prompt(
            self.repair_mode, object_reference, forced.candidate_box, result.reason
        )
        if self.repair_mode == 'blind_retry':
            raise ValueError(
                'one-shot reference repair requires binary_feedback, typed_feedback, '
                'concise_typed_feedback, separated_reference_feedback, or '
                'separated_reference_feedback_v2'
            )
        temporary_ids = self.tokenizer(repair_text, add_special_tokens=False).input_ids
        suppressed_eoc_indices = None
        if suppress_refbind_for_random_box:
            temporary_spans = find_coordinate_spans(
                temporary_ids, self.boc_token_id, self.eoc_token_id
            )
            if len(temporary_spans) != 1:
                raise RuntimeError(
                    'feature-only q-refbind ablation requires exactly one completed '
                    'coordinate in temporary repair feedback'
                )
            # H_t contains exactly the preceding completed coordinates; q is
            # the next generated EOC in this force-replayed sandbox.
            suppressed_eoc_indices = [len(find_coordinate_spans(
                h_t_ids, self.boc_token_id, self.eoc_token_id
            )) + 1]
        repaired = self._run_until_coordinate(
            h_t_ids, temporary_ids, max_new_tokens, temperature,
            suppress_refbind_generated_eoc_indices=suppressed_eoc_indices,
        )
        if repaired.candidate_span is None or repaired.candidate_box is None:
            raise RuntimeError('one-shot repair ended before a replacement coordinate')
        replacement_tokens = repaired.generated_ids[
            repaired.candidate_span[0]:repaired.candidate_span[1] + 1
        ]
        # The only commit: H_t + replacement.  ``_continue_freely`` replays it
        # from a clean prompt, thereby injecting V(replacement), then releases
        # all later coordinates and text without verifier calls.
        committed = h_t_ids + replacement_tokens
        continued_ids = self._continue_freely(committed, max_new_tokens, temperature)
        event = {
            'sample_id': self.sample_id,
            'mode': (
                'one_shot_reference_random_box_repair_keep_coor_skip_q_refbind'
                if suppress_refbind_for_random_box else 'one_shot_reference_random_box_repair'
            ),
            'source_coordinate_index': int(selected_coordinate_index),
            'object_reference': object_reference,
            'reference_prefix_text': self.tokenizer.decode(h_t_ids, skip_special_tokens=False),
            'reference_box': self._box_list(self._box_from_span(reference_generated_ids, target_span)),
            'random_box': self._box_list(forced.candidate_box),
            'forced_random_coordinate_text': self.tokenizer.decode(
                forced.generated_ids[forced.candidate_span[0]:forced.candidate_span[1] + 1],
                skip_special_tokens=False,
            ),
            'stored_oracle_check': {
                'lookup_key': {
                    'sample_id': self.sample_id,
                    'grounding_step': int(selected_coordinate_index),
                    'attempt_index': 0,
                },
                'candidate_bbox': self._box_list(forced.candidate_box),
                'verdict': result.verdict,
                'reason': result.reason,
                'confidence': result.confidence,
                'missing_oracle_record': lookup.missing_oracle_record,
                'oracle_candidate_mismatch': lookup.oracle_candidate_mismatch,
                'error': lookup.error,
            },
            'initial_verdict': result.verdict,
            'initial_reason': result.reason,
            'repair_prompt': repair_text,
            'sandbox_refbind_for_random_box': not suppress_refbind_for_random_box,
            'sandbox_coordinate_text_preserved': bool(suppress_refbind_for_random_box),
            'suppressed_refbind_generated_eoc_indices': suppressed_eoc_indices or [],
            'sandbox_bound_box_count_before_replacement': repaired.bound_box_count,
            'replacement_box': self._box_list(repaired.candidate_box),
            'replacement_coordinate_text': self.tokenizer.decode(
                replacement_tokens, skip_special_tokens=False,
            ),
            'repair_sandbox_text': self.tokenizer.decode(
                repaired.generated_ids, skip_special_tokens=False,
            ),
            'replacement_committed_without_second_verification': True,
            'later_coordinates_verified': False,
            'reference_prefix_replayed': True,
            'free_continuation_text': self.tokenizer.decode(
                continued_ids[len(committed):], skip_special_tokens=False,
            ),
        }
        self._write_event(event)
        return VerifierInferenceResult(
            self.tokenizer.decode(continued_ids, skip_special_tokens=False),
            continued_ids, 'ok', [event]
        )

    def run_one_shot_reference_repair_text_only_q(
            self, reference_generated_ids: Sequence[int],
            selected_coordinate_index: int, random_box: Sequence[float],
            max_new_tokens: int = 1024,
            temperature: float = 0.0) -> VerifierInferenceResult:
        """One-shot repair control: q is text-only and never binds in sandbox.

        This deliberately remains separate from the visual-aware repair method.
        Coordinates already present in H_t still follow ordinary REFbind during
        prefix replay; only the rejected q is not a valid coordinate span.
        """
        spans = find_coordinate_spans(reference_generated_ids, self.boc_token_id, self.eoc_token_id)
        selected_offset = int(selected_coordinate_index) - 1
        if not 0 <= selected_offset < len(spans):
            raise ValueError(f'selected_coordinate_index must be in [1, {len(spans)}]')
        target_span = spans[selected_offset]
        h_t_ids = list(reference_generated_ids[:target_span[0]])
        forced = self._run_forced_coordinate(h_t_ids, random_box, max_new_tokens, temperature)
        assert forced.candidate_box is not None and forced.candidate_span is not None
        lookup = self.verifier.verify(
            self.sample_id, int(selected_coordinate_index), 0, forced.candidate_box
        )
        result = lookup.result
        if (lookup.missing_oracle_record or lookup.oracle_candidate_mismatch
                or result.verdict != 'misaligned' or result.reason != 'wrong_object'):
            raise RuntimeError(
                'one-shot random intervention requires a matching stored '
                'misaligned/wrong_object oracle record'
            )

        object_reference = self._object_reference(h_t_ids)
        repair_text = build_repair_prompt_text_only_q(
            object_reference, forced.candidate_box, result.reason, self.repair_mode
        )
        temporary_ids = self.tokenizer(repair_text, add_special_tokens=False).input_ids
        repaired = self._run_until_coordinate(h_t_ids, temporary_ids, max_new_tokens, temperature)
        if repaired.candidate_span is None or repaired.candidate_box is None:
            raise RuntimeError('one-shot repair ended before a replacement coordinate')
        replacement_tokens = repaired.generated_ids[
            repaired.candidate_span[0]:repaired.candidate_span[1] + 1
        ]
        committed = h_t_ids + replacement_tokens
        continued_ids = self._continue_freely(committed, max_new_tokens, temperature)
        event = {
            'sample_id': self.sample_id,
            'mode': 'one_shot_reference_random_box_repair_text_only_q',
            'source_coordinate_index': int(selected_coordinate_index),
            'object_reference': object_reference,
            'reference_prefix_text': self.tokenizer.decode(h_t_ids, skip_special_tokens=False),
            'reference_box': self._box_list(self._box_from_span(reference_generated_ids, target_span)),
            'random_box': self._box_list(forced.candidate_box),
            'forced_random_coordinate_text': self.tokenizer.decode(
                forced.generated_ids[forced.candidate_span[0]:forced.candidate_span[1] + 1],
                skip_special_tokens=False,
            ),
            'stored_oracle_check': {
                'lookup_key': {
                    'sample_id': self.sample_id,
                    'grounding_step': int(selected_coordinate_index),
                    'attempt_index': 0,
                },
                'candidate_bbox': self._box_list(forced.candidate_box),
                'verdict': result.verdict,
                'reason': result.reason,
                'confidence': result.confidence,
                'missing_oracle_record': lookup.missing_oracle_record,
                'oracle_candidate_mismatch': lookup.oracle_candidate_mismatch,
                'error': lookup.error,
            },
            'initial_verdict': result.verdict,
            'initial_reason': result.reason,
            'repair_prompt': repair_text,
            'sandbox_refbind_for_random_box': False,
            'sandbox_expected_bound_box_count': len(find_coordinate_spans(
                h_t_ids, self.boc_token_id, self.eoc_token_id
            )),
            'sandbox_bound_box_count_before_replacement': repaired.bound_box_count,
            'replacement_box': self._box_list(repaired.candidate_box),
            'replacement_coordinate_text': self.tokenizer.decode(
                replacement_tokens, skip_special_tokens=False,
            ),
            'repair_sandbox_text': self.tokenizer.decode(
                repaired.generated_ids, skip_special_tokens=False,
            ),
            'replacement_committed_without_second_verification': True,
            'later_coordinates_verified': False,
            'reference_prefix_replayed': True,
            'free_continuation_text': self.tokenizer.decode(
                continued_ids[len(committed):], skip_special_tokens=False,
            ),
        }
        self._write_event(event)
        return VerifierInferenceResult(
            self.tokenizer.decode(continued_ids, skip_special_tokens=False),
            continued_ids, 'ok', [event]
        )

    def run_one_shot_reference_corruption(self, reference_generated_ids: Sequence[int],
                                          selected_coordinate_index: int, random_box: Sequence[float],
                                          max_new_tokens: int = 1024,
                                          temperature: float = 0.0) -> VerifierInferenceResult:
        """Commit a forced random box and freely continue without verification."""
        spans = find_coordinate_spans(reference_generated_ids, self.boc_token_id, self.eoc_token_id)
        selected_offset = int(selected_coordinate_index) - 1
        if not 0 <= selected_offset < len(spans):
            raise ValueError(f'selected_coordinate_index must be in [1, {len(spans)}]')
        target_span = spans[selected_offset]
        h_t_ids = list(reference_generated_ids[:target_span[0]])
        forced = self._run_forced_coordinate(h_t_ids, random_box, max_new_tokens, temperature)
        assert forced.candidate_span is not None and forced.candidate_box is not None
        # A clean replay commits q_i and consequently injects V(q_i) before
        # releasing the remaining CoT. This is the paired no-repair control.
        committed = h_t_ids + forced.generated_ids[forced.candidate_span[0]:forced.candidate_span[1] + 1]
        continued_ids = self._continue_freely(committed, max_new_tokens, temperature)
        event = {
            'sample_id': self.sample_id,
            'mode': 'one_shot_reference_random_box_corruption',
            'source_coordinate_index': int(selected_coordinate_index),
            'object_reference': self._object_reference(h_t_ids),
            'reference_box': self._box_list(self._box_from_span(reference_generated_ids, target_span)),
            'random_box': self._box_list(forced.candidate_box),
            'forced_random_coordinate_text': self.tokenizer.decode(
                forced.generated_ids[forced.candidate_span[0]:forced.candidate_span[1] + 1],
                skip_special_tokens=False,
            ),
            'random_box_committed': True,
            'random_box_visual_feature_injected': True,
            'verifier_called': False,
            'later_coordinates_verified': False,
            'reference_prefix_replayed': True,
            'free_continuation_text': self.tokenizer.decode(
                continued_ids[len(committed):], skip_special_tokens=False,
            ),
        }
        self._write_event(event)
        return VerifierInferenceResult(
            self.tokenizer.decode(continued_ids, skip_special_tokens=False),
            continued_ids, 'ok', [event]
        )

    def _accepted(self, lookup: VerificationLookup) -> bool:
        result = lookup.result
        return (
            not lookup.missing_oracle_record
            and not lookup.oracle_candidate_mismatch
            and result.verdict == 'aligned'
            and result.reason == 'none'
            and result.confidence >= self.accept_confidence
        )

    @staticmethod
    def _box_list(box: Optional[Box]):
        return None if box is None else [float(value) for value in box]

    def _write_event(self, event: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')

    def run(self, max_new_tokens: int = 1024, temperature: float = 0.0) -> VerifierInferenceResult:
        """Run free CoT while committing only oracle-verified coordinates."""
        if max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        persistent: List[int] = []
        events: List[Dict[str, Any]] = []
        grounding_step = 0

        while True:
            initial = self._run_until_coordinate(
                persistent, temporary_ids=[], max_new_tokens=max_new_tokens, temperature=temperature
            )
            if initial.candidate_span is None:
                response = self.tokenizer.decode(initial.generated_ids, skip_special_tokens=False)
                return VerifierInferenceResult(response, initial.generated_ids, 'ok', events)

            grounding_step += 1
            h_t_ids = initial.generated_ids[:initial.candidate_span[0]]
            # This is a textual snapshot.  The following clean replay recreates
            # visual state from accepted history; version one never stores KV.
            base_batch = self.batch_factory()
            prompt_ids = base_batch['input_ids'].reshape(-1).tolist()
            attention = base_batch['attention_mask'].reshape(-1).tolist()
            snapshot = RepairBaseSnapshot(
                repair_base_input_ids=tuple(prompt_ids + h_t_ids),
                repair_base_attention_mask=tuple(attention + [1] * len(h_t_ids)),
                repair_base_position=len(prompt_ids) + len(h_t_ids),
            )
            object_reference = self._object_reference(h_t_ids)
            candidate_box = initial.candidate_box
            assert candidate_box is not None and initial.candidate_span is not None
            started = time.perf_counter()
            lookup = self.verifier.verify(self.sample_id, grounding_step, 0, candidate_box)
            initial_result = lookup.result
            repair_attempts: List[Dict[str, Any]] = []
            committed_box: Optional[Box] = None
            accepted_tokens: Optional[List[int]] = None
            current_box = candidate_box
            current_lookup = lookup

            if self._accepted(lookup):
                committed_box = current_box
                accepted_tokens = initial.generated_ids[
                    initial.candidate_span[0]:initial.candidate_span[1] + 1
                ]
            else:
                for attempt_index in range(1, self.max_retries + 1):
                    repair_text = build_repair_prompt(
                        self.repair_mode, object_reference, current_box, current_lookup.result.reason
                    )
                    temporary_ids = ([] if self.repair_mode == 'blind_retry' else
                                     self.tokenizer(repair_text, add_special_tokens=False).input_ids)
                    if self.repair_mode != 'blind_retry' and not repair_text.endswith(DEFAULT_BOC_TOKEN):
                        raise RuntimeError('repair prompt must end with <coor>')
                    repaired = self._run_until_coordinate(
                        h_t_ids, temporary_ids, max_new_tokens=max_new_tokens, temperature=temperature
                    )
                    if repaired.candidate_span is None or repaired.candidate_box is None:
                        raise RuntimeError('repair generation ended before a replacement </coor>')
                    replacement = repaired.candidate_box
                    replacement_lookup = self.verifier.verify(
                        self.sample_id, grounding_step, attempt_index, replacement
                    )
                    repair_attempts.append({
                        'attempt_index': attempt_index,
                        'rejected_bbox_in_prompt': self._box_list(current_box),
                        'generated_bbox': self._box_list(replacement),
                        'verdict': replacement_lookup.result.verdict,
                        'reason': replacement_lookup.result.reason,
                        'confidence': replacement_lookup.result.confidence,
                        'missing_oracle_record': replacement_lookup.missing_oracle_record,
                        'oracle_candidate_mismatch': replacement_lookup.oracle_candidate_mismatch,
                    })
                    if self._accepted(replacement_lookup):
                        committed_box = replacement
                        accepted_tokens = repaired.generated_ids[
                            repaired.candidate_span[0]:repaired.candidate_span[1] + 1
                        ]
                        break
                    current_box, current_lookup = replacement, replacement_lookup

            succeeded = committed_box is not None and accepted_tokens is not None
            event = {
                'sample_id': self.sample_id,
                'grounding_step': grounding_step,
                'object_reference': object_reference,
                'h_t_ends_before_coor': True,
                'repair_base_input_id_count': len(snapshot.repair_base_input_ids),
                'repair_base_attention_mask_count': len(snapshot.repair_base_attention_mask),
                'repair_base_position': snapshot.repair_base_position,
                'repair_base_kv_cache': None,
                'initial_bbox': self._box_list(candidate_box),
                'initial_verdict': initial_result.verdict,
                'initial_reason': initial_result.reason,
                'initial_confidence': initial_result.confidence,
                'initial_missing_oracle_record': lookup.missing_oracle_record,
                'initial_oracle_candidate_mismatch': lookup.oracle_candidate_mismatch,
                'repair_mode': self.repair_mode,
                'repair_attempts': repair_attempts,
                'committed_bbox': self._box_list(committed_box),
                'repair_success': succeeded,
                'rejected_coor_in_persistent_context': False,
                'feedback_in_persistent_context': False,
                'visual_feature_injected': succeeded,
                'temporary_rejected_visual_feature_injected': bool(
                    repair_attempts and self.repair_mode != 'blind_retry'
                ),
                'latency_ms': round((time.perf_counter() - started) * 1000, 3),
            }
            events.append(event)
            if not succeeded:
                if self.on_failure == 'abort_sample':
                    event['failure_action'] = 'abort_sample'
                    self._write_event(event)
                    response = self.tokenizer.decode(h_t_ids, skip_special_tokens=False)
                    return VerifierInferenceResult(response, h_t_ids, 'repair_failed', events)
                # Remove this object's immediate grounding decision, then let
                # the original generator freely revise the remaining CoT. The
                # processor masks only that first possible <coor>, not all
                # later coordinates in the revised trajectory.
                continued_ids, first_boc_suppressed = self._continue_after_failed_grounding(
                    h_t_ids, max_new_tokens=max_new_tokens, temperature=temperature
                )
                event['failure_action'] = 'skip_current_grounding_and_continue'
                event['first_post_h_t_boc_suppressed'] = first_boc_suppressed
                event['latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
                self._write_event(event)
                response = self.tokenizer.decode(continued_ids, skip_special_tokens=False)
                return VerifierInferenceResult(
                    response, continued_ids, 'repair_failed_skip_grounding_continued', events
                )

            self._write_event(event)

            # Commit with H_t, not the temporary repair context.  The next
            # iteration performs a clean prefill and lets native generate_box
            # inject this accepted feature when replaying its closing EOC.
            persistent = h_t_ids + accepted_tokens
