"""Utilities for token-level counterfactual coordinate interventions."""

import math
import random
import re
from typing import Any, Dict, List, Sequence, Tuple

from transformers import LogitsProcessor

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN


Box = Tuple[float, float, float, float]


def normalized_box_to_square_padding(
    box: Sequence[float],
    image_width: int,
    image_height: int,
) -> Box:
    """Map an original-image box onto VoCoT's center-padded square canvas.

    ``VoCoT_InputProcessor.expand2square_fn`` pastes the original image at an
    integer ``// 2`` offset on the short axis.  Coordinates consumed by
    ``generate_box``/REFbind are normalized against that padded square, so GT
    boxes normalized against the unpadded image must undergo the same mapping.
    """
    if len(box) != 4:
        raise ValueError('box must contain xmin, ymin, xmax, ymax')
    if (
        not isinstance(image_width, int)
        or isinstance(image_width, bool)
        or not isinstance(image_height, int)
        or isinstance(image_height, bool)
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError(
            f'image dimensions must be positive integers, got {image_width}x{image_height}'
        )
    original = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in original):
        raise ValueError(f'box must be finite: {box}')
    x_min, y_min, x_max, y_max = original
    if not 0 <= x_min < x_max <= 1 or not 0 <= y_min < y_max <= 1:
        raise ValueError(f'invalid normalized xyxy box: {box}')

    square_size = max(image_width, image_height)
    absolute = [
        x_min * image_width,
        y_min * image_height,
        x_max * image_width,
        y_max * image_height,
    ]
    if image_width > image_height:
        padding = (image_width - image_height) // 2
        absolute[1] += padding
        absolute[3] += padding
    elif image_height > image_width:
        padding = (image_height - image_width) // 2
        absolute[0] += padding
        absolute[2] += padding
    return tuple(value / square_size for value in absolute)


def normalize_object_reference(text: str) -> Tuple[str, ...]:
    """Normalize an object reference for strict, token-level alias matching.

    This intentionally preserves descriptive modifiers such as colours and
    materials.  It only removes surface variation that does not identify a
    different object: case, punctuation, hyphenation, articles, and possessive
    suffixes.  The result is a token tuple so ``car`` never matches ``cart``.
    """
    if not isinstance(text, str):
        return ()
    text = text.lower().replace('’', "'")
    text = re.sub(r"([a-z0-9])'s\b", r"\1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return tuple(
        token for token in text.split()
        if token not in {'a', 'an', 'the'}
    )


def _validate_oracle_box(box: Sequence[float], precision: int) -> Box:
    if len(box) != 4:
        raise ValueError('oracle box must contain xmin, ymin, xmax, ymax')
    rounded = tuple(round(float(value), precision) for value in box)
    if not all(math.isfinite(value) for value in rounded):
        raise ValueError(f'oracle box must be finite: {box}')
    x_min, y_min, x_max, y_max = rounded
    if not 0 <= x_min < x_max <= 1 or not 0 <= y_min < y_max <= 1:
        raise ValueError(
            f'oracle box becomes invalid after {precision}-decimal quantization: {box} -> {rounded}'
        )
    return rounded


def _find_token_subsequence_positions(tokens: Sequence[str], needle: Sequence[str]) -> List[int]:
    if not needle or len(needle) > len(tokens):
        return []
    width = len(needle)
    return [
        start for start in range(len(tokens) - width + 1)
        if tuple(tokens[start:start + width]) == tuple(needle)
    ]


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


class PrefixReplayRemoveGroundingLogitsProcessor(LogitsProcessor):
    """Replay through a selected point, then suppress that ``<coor>`` token.

    The processor replays only the baseline tokens *before* the selected
    opening coordinate tag.  At the original tag position it masks ``<coor>``
    and immediately releases all later decoding.  Since neither ``<coor>`` nor
    its matching ``</coor>`` is generated at that position, VoCoT never calls
    ``generate_box`` and no REFbind feature is injected for the removed
    grounding.  A later coordinate generated freely by the model is retained
    as a new post-intervention reasoning decision.
    """

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        baseline_generated_ids: Sequence[int],
        intervention_boc_offset: int,
    ):
        if not 0 <= intervention_boc_offset < len(baseline_generated_ids):
            raise ValueError('intervention_boc_offset is outside the baseline generation')
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.baseline_generated_ids = list(baseline_generated_ids)
        self.intervention_boc_offset = intervention_boc_offset
        self.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
        if self.baseline_generated_ids[intervention_boc_offset] != self.boc_token_id:
            raise ValueError('intervention_boc_offset must identify a <coor> token')
        self.suppressed_boc = False
        self.released = False

    @staticmethod
    def _force_token(scores, token_id: int):
        scores.fill_(float('-inf'))
        scores[:, token_id] = 0
        return scores

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError('PrefixReplayRemoveGroundingLogitsProcessor supports batch size 1 only')
        generated_ids = input_ids[0, self.prompt_length:].tolist()

        # Do not replay the selected <coor> itself.  Earlier coordinate spans
        # close normally and therefore retain their original REFbind features.
        if len(generated_ids) < self.intervention_boc_offset:
            expected_prefix = self.baseline_generated_ids[:len(generated_ids)]
            if generated_ids != expected_prefix:
                raise RuntimeError('counterfactual generation diverged before grounding removal')
            return self._force_token(scores, self.baseline_generated_ids[len(generated_ids)])

        if len(generated_ids) == self.intervention_boc_offset:
            expected_prefix = self.baseline_generated_ids[:self.intervention_boc_offset]
            if generated_ids != expected_prefix:
                raise RuntimeError('counterfactual prefix diverged before grounding removal')
            scores[:, self.boc_token_id] = float('-inf')
            self.suppressed_boc = True
            self.released = True
            return scores

        # The first free token must not be the suppressed opening tag.  From
        # then on the model is completely free, including to ground a newly
        # planned region later in its revised CoT.
        if generated_ids[self.intervention_boc_offset] == self.boc_token_id:
            raise RuntimeError('selected <coor> was not suppressed')
        return scores


class OnlineOracleCoordinateLogitsProcessor(LogitsProcessor):
    """Correct explicitly referenced target boxes during free CoT decoding.

    The model still chooses when to emit ``<coor>`` and generates all ordinary
    CoT tokens freely.  Immediately after an opening coordinate tag, this
    processor examines only the preceding local text.  If that text contains a
    unique, strict alias of a supplied oracle target, it forces that target's
    ground-truth coordinate text and closing tag.  Otherwise it leaves logits
    untouched and the model generates its own coordinate and visual binding.

    No baseline tokens are replayed.  Thus every forced coordinate can affect
    all following CoT tokens through VoCoT's normal ``generate_box`` pathway.
    """

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        oracle_targets: Sequence[Dict[str, Any]],
        precision: int = 3,
        context_window_tokens: int = 48,
    ):
        if prompt_length < 0:
            raise ValueError('prompt_length must be non-negative')
        if precision < 0:
            raise ValueError('precision must be non-negative')
        if context_window_tokens <= 0:
            raise ValueError('context_window_tokens must be positive')
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.precision = precision
        self.context_window_tokens = context_window_tokens
        self.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
        self.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
        self.targets = self._prepare_targets(oracle_targets)
        self._decisions_by_boc_offset: Dict[int, Dict[str, Any]] = {}
        self._suffix_by_boc_offset: Dict[int, List[int]] = {}

    def _prepare_targets(self, oracle_targets: Sequence[Dict[str, Any]]):
        prepared = []
        seen_aliases = {}
        for target_index, target in enumerate(oracle_targets):
            if not isinstance(target, dict):
                raise TypeError('each oracle target must be a dictionary')
            object_name = target.get('object')
            box = target.get('box')
            aliases = target.get('aliases', [object_name])
            if not isinstance(object_name, str) or not object_name.strip():
                raise ValueError('oracle target requires a non-empty object name')
            if not isinstance(aliases, (list, tuple)) or not aliases:
                raise ValueError(f'oracle target {object_name!r} requires at least one alias')
            normalized_aliases = []
            for alias in aliases:
                normalized = normalize_object_reference(alias)
                if not normalized:
                    raise ValueError(f'invalid empty alias for oracle target {object_name!r}')
                if normalized not in normalized_aliases:
                    normalized_aliases.append(normalized)
            rounded_box = _validate_oracle_box(box, self.precision)
            for alias in normalized_aliases:
                if alias in seen_aliases:
                    previous = seen_aliases[alias]
                    raise ValueError(
                        f'ambiguous alias {" ".join(alias)!r} shared by targets '
                        f'{previous!r} and {object_name!r}'
                    )
                seen_aliases[alias] = object_name
            prepared.append({
                'target_index': target_index,
                'object': object_name,
                'box': rounded_box,
                'aliases': normalized_aliases,
            })
        if not prepared:
            raise ValueError('online oracle requires at least one target')
        return prepared

    def _context_before_coordinate(self, generated_ids: Sequence[int], boc_offset: int):
        """Return the latest free-text segment and its normalized tail tokens."""
        prefix_text = self.tokenizer.decode(
            generated_ids[:boc_offset], skip_special_tokens=False
        )
        # A new coordinate should be linked only to text since the preceding
        # closed coordinate, never to an entity mentioned arbitrarily early in
        # the rollout.
        local_text = prefix_text.rsplit(DEFAULT_EOC_TOKEN, 1)[-1]
        normalized_tokens = normalize_object_reference(local_text)
        return local_text, normalized_tokens[-self.context_window_tokens:]

    def _match_target(self, local_text: str, context_tokens: Sequence[str]):
        """Return a unique most-recent, longest explicit alias match, if any."""
        candidates = []
        for target in self.targets:
            for alias in target['aliases']:
                for start in _find_token_subsequence_positions(context_tokens, alias):
                    candidates.append({
                        'target_index': target['target_index'],
                        'object': target['object'],
                        'box': target['box'],
                        'alias_tokens': alias,
                        'start': start,
                        'end': start + len(alias),
                    })
        if not candidates:
            return None, 'no_explicit_target_alias'

        # The immediate referent is the latest explicit object phrase.  A
        # longer phrase wins only when phrases end at the same position.
        latest_end = max(candidate['end'] for candidate in candidates)
        latest_candidates = [
            candidate for candidate in candidates if candidate['end'] == latest_end
        ]
        longest_length = max(len(candidate['alias_tokens']) for candidate in latest_candidates)
        best = [
            candidate for candidate in latest_candidates
            if len(candidate['alias_tokens']) == longest_length
        ]
        target_indices = {candidate['target_index'] for candidate in best}
        if len(target_indices) != 1:
            return None, 'ambiguous_explicit_target_alias'
        return best[0], 'explicit_alias'

    def _make_suffix_ids(self, box: Box):
        box_text = ','.join(f'{value:.{self.precision}f}' for value in box)
        suffix_text = f'{box_text}{DEFAULT_EOC_TOKEN}'
        suffix_ids = self.tokenizer(suffix_text, add_special_tokens=False).input_ids
        if not suffix_ids or suffix_ids[-1] != self.eoc_token_id:
            raise ValueError(f'could not tokenize oracle coordinate suffix: {suffix_text}')
        return suffix_ids

    def _new_decision(self, generated_ids: Sequence[int], boc_offset: int, coordinate_index: int):
        local_text, context_tokens = self._context_before_coordinate(generated_ids, boc_offset)
        matched, match_tier = self._match_target(local_text, context_tokens)
        decision = {
            'coordinate_index': coordinate_index,
            'boc_generated_offset': boc_offset,
            'context': local_text[-400:],
            'context_normalized_tokens': list(context_tokens),
            'decision': 'kept_model_box',
            'reason': match_tier,
            'target_object': None,
            'matched_alias': None,
            'oracle_box': None,
        }
        if matched is not None:
            decision.update({
                'decision': 'forced_gt_box',
                'target_object': matched['object'],
                'matched_alias': ' '.join(matched['alias_tokens']),
                'oracle_box': list(matched['box']),
            })
            self._suffix_by_boc_offset[boc_offset] = self._make_suffix_ids(matched['box'])
        self._decisions_by_boc_offset[boc_offset] = decision
        return decision

    @property
    def events(self):
        return [
            self._decisions_by_boc_offset[offset]
            for offset in sorted(self._decisions_by_boc_offset)
        ]

    @staticmethod
    def _force_token(scores, token_id: int):
        scores.fill_(float('-inf'))
        scores[:, token_id] = 0
        return scores

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError('OnlineOracleCoordinateLogitsProcessor supports batch size 1 only')
        generated_ids = input_ids[0, self.prompt_length:].tolist()
        boc_positions = [
            offset for offset, token_id in enumerate(generated_ids)
            if token_id == self.boc_token_id
        ]
        eoc_positions = [
            offset for offset, token_id in enumerate(generated_ids)
            if token_id == self.eoc_token_id
        ]
        if not boc_positions or (eoc_positions and eoc_positions[-1] > boc_positions[-1]):
            return scores

        current_boc = boc_positions[-1]
        coordinate_index = len(eoc_positions) + 1
        if current_boc not in self._decisions_by_boc_offset:
            self._new_decision(generated_ids, current_boc, coordinate_index)
        suffix_ids = self._suffix_by_boc_offset.get(current_boc)
        if suffix_ids is None:
            return scores

        observed_suffix = generated_ids[current_boc + 1:]
        expected_prefix = suffix_ids[:len(observed_suffix)]
        if observed_suffix != expected_prefix:
            raise RuntimeError('generated coordinate diverged from the forced oracle suffix')
        if len(observed_suffix) >= len(suffix_ids):
            return scores
        return self._force_token(scores, suffix_ids[len(observed_suffix)])
