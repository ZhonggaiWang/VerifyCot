"""Conservative reference-to-GT resolution for oracle-only experiments.

This module is deliberately neutral infrastructure rather than a verifier or
an expert backend.  Oracle verifiers, Grounders, and BoxRefiners all consume
the same resolver so their target identity policy cannot silently diverge.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from constants import DEFAULT_EOC_TOKEN
from utils.coordinate_intervention import (
    ExplicitOracleTargetMatcher,
    normalize_object_reference,
)

from .contracts import VerificationRequest
from .types import Box


ORACLE_TARGET_MATCH_POLICY = ExplicitOracleTargetMatcher.POLICY


@dataclass(frozen=True)
class OracleTargetResolution:
    """Result of resolving one local generated reference against GT targets."""

    matched: bool
    reason: str
    target_object: Optional[str] = None
    matched_alias: Optional[str] = None
    bbox: Optional[Box] = None
    context: str = ''
    context_normalized_tokens: Tuple[str, ...] = ()
    target_index: Optional[int] = None

    def as_metadata(self) -> Dict[str, Any]:
        return {
            'oracle_resolution_matched': self.matched,
            'oracle_resolution_reason': self.reason,
            'oracle_resolution_policy': ORACLE_TARGET_MATCH_POLICY,
            'target_object': self.target_object,
            'matched_alias': self.matched_alias,
            'oracle_target_box': (
                None if self.bbox is None else list(self.bbox)
            ),
            'oracle_resolution_context': self.context,
            'oracle_resolution_context_normalized_tokens': list(
                self.context_normalized_tokens
            ),
            'oracle_target_index': self.target_index,
        }


class OracleTargetResolver:
    """Resolve the latest unique explicit target alias before ``<coor>``.

    It examines only generated text after the previous completed coordinate.
    Pronouns and implicit coreference are never guessed.  Exact duplicate GT
    records are merged by the shared matcher; one alias shared by different
    target instances is an ambiguous non-match.
    """

    def __init__(
            self,
            tokenizer,
            oracle_targets: Optional[Sequence[Dict[str, Any]]] = None,
            *,
            context_window_tokens: int = 48,
            precision: int = 3):
        if int(context_window_tokens) <= 0:
            raise ValueError('context_window_tokens must be positive')
        if int(precision) < 0:
            raise ValueError('precision must be non-negative')
        self.tokenizer = tokenizer
        self.context_window_tokens = int(context_window_tokens)
        self.precision = int(precision)
        self._fixed_targets = (
            None if oracle_targets is None else tuple(oracle_targets)
        )
        self._fixed_matcher = (
            None
            if self._fixed_targets is None
            else ExplicitOracleTargetMatcher(
                self._fixed_targets,
                precision=self.precision,
            )
        )

    def _matcher_for_request(
            self,
            request: VerificationRequest) -> Optional[ExplicitOracleTargetMatcher]:
        if self._fixed_matcher is not None:
            return self._fixed_matcher
        targets = request.sample_context.get('oracle_targets')
        if not isinstance(targets, (list, tuple)) or not targets:
            return None
        return ExplicitOracleTargetMatcher(
            targets,
            precision=self.precision,
        )

    def _local_context(
            self,
            request: VerificationRequest) -> Tuple[str, Tuple[str, ...]]:
        boc_offset = int(request.candidate_span[0])
        if not 0 <= boc_offset < len(request.generated_ids):
            raise ValueError(
                'candidate BOC offset is outside generated_ids: '
                f'{boc_offset} vs {len(request.generated_ids)} tokens'
            )
        prefix_text = self.tokenizer.decode(
            request.generated_ids[:boc_offset],
            skip_special_tokens=False,
        )
        local_text = prefix_text.rsplit(DEFAULT_EOC_TOKEN, 1)[-1]
        normalized = normalize_object_reference(local_text)
        return (
            local_text,
            normalized[-self.context_window_tokens:],
        )

    def resolve(
            self,
            request: VerificationRequest) -> OracleTargetResolution:
        matcher = self._matcher_for_request(request)
        if matcher is None:
            return OracleTargetResolution(
                matched=False,
                reason='missing_oracle_targets',
            )

        local_text, context_tokens = self._local_context(request)
        matched, reason = matcher.match(context_tokens)
        common = {
            'reason': str(reason),
            'context': local_text[-400:],
            'context_normalized_tokens': tuple(context_tokens),
        }
        if matched is None:
            return OracleTargetResolution(
                matched=False,
                **common,
            )
        return OracleTargetResolution(
            matched=True,
            target_object=str(matched['object']),
            matched_alias=' '.join(matched['alias_tokens']),
            bbox=matched['box'],
            target_index=int(matched['target_index']),
            **common,
        )


__all__ = [
    'ORACLE_TARGET_MATCH_POLICY',
    'OracleTargetResolution',
    'OracleTargetResolver',
]
