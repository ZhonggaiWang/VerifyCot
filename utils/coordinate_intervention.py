"""Utilities for token-level counterfactual coordinate interventions."""

import random
from typing import Sequence, Tuple

from transformers import LogitsProcessor

from constants import DEFAULT_EOC_TOKEN


Box = Tuple[float, float, float, float]


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return IoU for normalized ``[xmin, ymin, xmax, ymax]`` boxes."""
    x_min = max(first[0], second[0])
    y_min = max(first[1], second[1])
    x_max = min(first[2], second[2])
    y_max = min(first[3], second[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def make_same_shape_perturbation(
    box: Sequence[float],
    rng: random.Random,
    iou_range: Tuple[float, float] = (0.0, 0.2),
    precision: int = 3,
    max_trials: int = 1000,
) -> Box:
    """Translate a box while preserving its size and targeting an IoU interval."""
    if len(box) != 4:
        raise ValueError("box must contain xmin, ymin, xmax, ymax")
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    width, height = x_max - x_min, y_max - y_min
    iou_min, iou_max = iou_range
    if not 0 <= iou_min <= iou_max <= 1:
        raise ValueError("iou_range must satisfy 0 <= min <= max <= 1")
    if width <= 0 or height <= 0 or width > 1 or height > 1:
        raise ValueError(f"invalid source box: {box}")

    for _ in range(max_trials):
        new_x_min = rng.uniform(0.0, 1.0 - width)
        new_y_min = rng.uniform(0.0, 1.0 - height)
        candidate = tuple(round(value, precision) for value in (
            new_x_min,
            new_y_min,
            new_x_min + width,
            new_y_min + height,
        ))
        candidate_iou = box_iou(box, candidate)
        if iou_min <= candidate_iou <= iou_max:
            return candidate
    raise RuntimeError(
        f"could not sample a same-shape perturbation in IoU range {iou_range} "
        f"for box {box} after {max_trials} trials"
    )


def make_random_box_perturbation(
    box: Sequence[float],
    rng: random.Random,
    iou_range: Tuple[float, float] = (0.0, 0.2),
    min_box_size: float = 0.05,
    max_box_size: float = 0.5,
    precision: int = 3,
    max_trials: int = 1000,
) -> Box:
    """Sample an erroneous box with independently randomized size and location.

    Unlike ``make_same_shape_perturbation``, this can perturb near-full-image
    boxes.  Restricting the sampled side lengths keeps the replacement box
    sufficiently different even when the source box covers most of the image.
    """
    if len(box) != 4:
        raise ValueError("box must contain xmin, ymin, xmax, ymax")
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    if not 0 <= x_min < x_max <= 1 or not 0 <= y_min < y_max <= 1:
        raise ValueError(f"invalid source box: {box}")
    iou_min, iou_max = iou_range
    if not 0 <= iou_min <= iou_max <= 1:
        raise ValueError("iou_range must satisfy 0 <= min <= max <= 1")
    if not 0 < min_box_size <= max_box_size <= 1:
        raise ValueError("box sizes must satisfy 0 < min_box_size <= max_box_size <= 1")

    for _ in range(max_trials):
        width = rng.uniform(min_box_size, max_box_size)
        height = rng.uniform(min_box_size, max_box_size)
        new_x_min = rng.uniform(0.0, 1.0 - width)
        new_y_min = rng.uniform(0.0, 1.0 - height)
        candidate = tuple(round(value, precision) for value in (
            new_x_min,
            new_y_min,
            new_x_min + width,
            new_y_min + height,
        ))
        candidate_iou = box_iou(box, candidate)
        if iou_min <= candidate_iou <= iou_max:
            return candidate
    raise RuntimeError(
        f"could not sample a random-size perturbation in IoU range {iou_range} "
        f"for box {box} after {max_trials} trials"
    )


def find_coordinate_spans(token_ids: Sequence[int], boc_token_id: int, eoc_token_id: int):
    """Return inclusive ``(boc_offset, eoc_offset)`` spans in generated ids."""
    spans = []
    open_boc = None
    for offset, token_id in enumerate(token_ids):
        if token_id == boc_token_id:
            open_boc = offset
        elif token_id == eoc_token_id and open_boc is not None:
            spans.append((open_boc, offset))
            open_boc = None
    return spans


class PrefixReplayCoordinateLogitsProcessor(LogitsProcessor):
    """Replay a baseline prefix, replace one coordinate, then release decoding."""

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        baseline_generated_ids: Sequence[int],
        intervention_boc_offset: int,
        replacement_box: Sequence[float],
        precision: int = 3,
    ):
        if not 0 <= intervention_boc_offset < len(baseline_generated_ids):
            raise ValueError("intervention_boc_offset is outside the baseline generation")
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.baseline_generated_ids = list(baseline_generated_ids)
        self.intervention_boc_offset = intervention_boc_offset
        self.replacement_box = tuple(round(float(value), precision) for value in replacement_box)
        self.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
        box_text = ",".join(f"{value:.{precision}f}" for value in self.replacement_box)
        self.replacement_suffix_ids = tokenizer(
            f"{box_text}{DEFAULT_EOC_TOKEN}", add_special_tokens=False
        ).input_ids
        if not self.replacement_suffix_ids or self.replacement_suffix_ids[-1] != self.eoc_token_id:
            raise ValueError(f"could not tokenize replacement box: {box_text}")
        self.released = False

    @staticmethod
    def _force_token(scores, token_id: int):
        scores.fill_(float("-inf"))
        scores[:, token_id] = 0
        return scores

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError("PrefixReplayCoordinateLogitsProcessor supports batch size 1 only")
        generated_ids = input_ids[0, self.prompt_length:].tolist()

        # Replay every baseline token up to and including the intervention's
        # <coor>.  Earlier </coor> tokens therefore take the native bind path.
        if len(generated_ids) <= self.intervention_boc_offset:
            expected_prefix = self.baseline_generated_ids[:len(generated_ids)]
            if generated_ids != expected_prefix:
                raise RuntimeError("counterfactual generation diverged before the intervention")
            return self._force_token(scores, self.baseline_generated_ids[len(generated_ids)])

        replacement_start = self.intervention_boc_offset + 1
        observed_suffix = generated_ids[replacement_start:]
        if observed_suffix[:len(self.replacement_suffix_ids)] != self.replacement_suffix_ids[:len(observed_suffix)]:
            raise RuntimeError("counterfactual generation diverged inside the replacement coordinate")
        if len(observed_suffix) < len(self.replacement_suffix_ids):
            return self._force_token(scores, self.replacement_suffix_ids[len(observed_suffix)])

        self.released = True
        return scores
