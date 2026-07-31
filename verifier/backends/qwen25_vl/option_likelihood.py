"""Model-agnostic selection from single-token option likelihoods.

The Qwen runner owns tokenization and model execution.  This module only
defines the scoring contract and converts negative log-likelihoods into one
deterministic classification decision, keeping benchmark and routing code
independent from Qwen internals.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class SingleTokenOptionScores:
    """Auditable next-token negative log-likelihoods for named options."""

    negative_log_likelihoods: Dict[str, float]
    token_ids: Dict[str, int]

    def __post_init__(self) -> None:
        labels = tuple(self.negative_log_likelihoods)
        if not labels:
            raise ValueError('at least one option score is required')
        if set(labels) != set(self.token_ids):
            raise ValueError('option score labels and token-id labels must match')
        if any(
            not math.isfinite(float(value))
            for value in self.negative_log_likelihoods.values()
        ):
            raise ValueError('option negative log-likelihoods must be finite')
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in self.token_ids.values()
        ):
            raise ValueError('option token ids must be non-negative integers')


class SingleTokenOptionLikelihoodRunner(Protocol):
    """Score fixed one-token completions after one multimodal prompt."""

    def score_single_token_options(
            self,
            messages: Sequence[Mapping[str, Any]],
            options: Mapping[str, str],
    ) -> SingleTokenOptionScores:
        raise NotImplementedError


@dataclass(frozen=True)
class OptionLikelihoodDecision:
    """Lowest-NLL option plus normalized audit statistics."""

    label: str
    confidence: float
    nll_margin: float
    negative_log_likelihoods: Dict[str, float]
    normalized_probabilities: Dict[str, float]
    token_ids: Dict[str, int]


def decide_from_option_scores(
        scores: SingleTokenOptionScores,
) -> OptionLikelihoodDecision:
    """Select the lowest-NLL option and normalize probability within options."""

    losses = {
        str(label): float(value)
        for label, value in scores.negative_log_likelihoods.items()
    }
    winner = min(losses, key=losses.__getitem__)
    minimum = losses[winner]
    unnormalized = {
        label: math.exp(-(loss - minimum))
        for label, loss in losses.items()
    }
    denominator = sum(unnormalized.values())
    probabilities = {
        label: value / denominator
        for label, value in unnormalized.items()
    }
    ordered_losses = sorted(losses.values())
    margin = (
        ordered_losses[1] - ordered_losses[0]
        if len(ordered_losses) > 1
        else math.inf
    )
    return OptionLikelihoodDecision(
        label=winner,
        confidence=probabilities[winner],
        nll_margin=margin,
        negative_log_likelihoods=losses,
        normalized_probabilities=probabilities,
        token_ids=dict(scores.token_ids),
    )
