"""Temporary repair prompts.  These strings are never committed to CoT."""

from typing import Literal, Sequence

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN


RepairMode = Literal[
    'blind_retry', 'binary_feedback', 'typed_feedback', 'concise_typed_feedback',
    'separated_reference_feedback'
]


def format_box(box: Sequence[float], precision: int = 3) -> str:
    return ','.join(f'{float(value):.{precision}f}' for value in box)


def build_repair_prompt(repair_mode: RepairMode, object_reference: str,
                        rejected_bbox: Sequence[float], reason: str) -> str:
    """Return a temporary prompt ending exactly with an opening ``<coor>``."""
    if repair_mode == 'blind_retry':
        return ''
    if repair_mode not in {
        'binary_feedback', 'typed_feedback', 'concise_typed_feedback',
        'separated_reference_feedback'
    }:
        raise ValueError(f'unknown repair_mode: {repair_mode}')
    coordinate = f'{DEFAULT_BOC_TOKEN}{format_box(rejected_bbox)}{DEFAULT_EOC_TOKEN}'
    ref = object_reference.strip() or 'the current object reference'
    if repair_mode == 'binary_feedback':
        body = (
            f'Previous reasoning: {ref}{coordinate}. '
            'This coordinate was rejected because it does not reliably localize the '
            'referenced object.\n\n'
            f'I will keep "{ref}" unchanged, re-reason about the same object, and '
            'select a new coordinate without reusing the rejected region.\n\n'
            # Keep the final generation boundary in VoCoT's training form:
            # object reference immediately followed by <coor>, with no quote,
            # punctuation, or explanatory token between them.
            f'I re-reason as {ref}{DEFAULT_BOC_TOKEN}'
        )
    elif repair_mode == 'typed_feedback':
        templates = {
            'wrong_object': (
                f'Previous reasoning: {ref}{coordinate}. '
                'This coordinate was rejected because it localizes a different object.\n\n'
                f'I will keep "{ref}" unchanged and re-reason about the same object. '
                'I will relocate it to a different region, avoiding reuse of the rejected '
                'region and preferably using low overlap with it.\n\n'
                f'I re-reason as {ref}{DEFAULT_BOC_TOKEN}'
            ),
            'partial_coverage': (
                f'Previous reasoning: {ref}{coordinate}. '
                'This coordinate was rejected because it covers only part of the object.\n\n'
                f'I will keep "{ref}" unchanged and re-reason about the same object. '
                'I will adjust or expand the region to cover the complete object while '
                'excluding unnecessary unrelated regions.\n\n'
                f'I re-reason as {ref}{DEFAULT_BOC_TOKEN}'
            ),
            'ambiguous': (
                f'Previous reasoning: {ref}{coordinate}. '
                'This coordinate was rejected because it is ambiguous and contains '
                'multiple possible objects.\n\n'
                f'I will keep "{ref}" unchanged and re-reason about the same object. '
                'I will select a tighter, more discriminative region that uniquely '
                'localizes it.\n\n'
                f'I re-reason as {ref}{DEFAULT_BOC_TOKEN}'
            ),
        }
        body = templates.get(reason, (
            f'Previous reasoning: {ref}{coordinate}. '
            'This coordinate could not be verified reliably.\n\n'
            f'I will keep "{ref}" unchanged and re-reason about the same object. '
            'I will generate a clearer region that uniquely localizes it.\n\n'
            f'I re-reason as {ref}{DEFAULT_BOC_TOKEN}'
        ))
    elif repair_mode == 'concise_typed_feedback':
        # A compact alternative for the one-shot experiment.  The old
        # ``typed_feedback`` wording is deliberately retained above as the
        # original baseline.  Here, the rejected coordinate and the next
        # generation boundary are separated by a short, explicit reset.  The
        # final line still follows VoCoT's native ``object<coor>`` pattern,
        # rather than the less familiar "I re-reason as object<coor>" form.
        concise_reasons = {
            'wrong_object': (
                'Verification: rejected — this region localizes a different object. '
                'Do not reuse this region.\n\n'
                'Recheck the image and localize the same object from scratch.\n'
            ),
            'partial_coverage': (
                'Verification: rejected — this region covers only part of the object. '
                'Do not reuse this region unchanged.\n\n'
                'Recheck the image and localize the complete same object.\n'
            ),
            'ambiguous': (
                'Verification: rejected — this region is ambiguous. '
                'Do not reuse this region.\n\n'
                'Recheck the image and localize only the same object.\n'
            ),
        }
        body = (
            f'Previous reasoning: {ref}{coordinate}\n'
            + concise_reasons.get(reason, (
                'Verification: rejected — this region cannot be verified reliably. '
                'Do not reuse this region.\n\n'
                'Recheck the image and localize the same object from scratch.\n'
            ))
            # This must be the final, directly decodable boundary.
            + f'{ref}{DEFAULT_BOC_TOKEN}'
        )
    else:
        # Keep the rejected coordinate in the sandbox so Volcano binds V(q),
        # but never expose it in the trained ``object<coor>q</coor>`` pattern.
        # Only the final decoding boundary uses that pattern for the new box.
        separated_reasons = {
            'wrong_object': (
                'Verification: rejected — it localizes a different object. '
                'Do not reuse that region.\n\n'
            ),
            'partial_coverage': (
                'Verification: rejected — it covers only part of the object. '
                'Do not reuse that region unchanged.\n\n'
            ),
            'ambiguous': (
                'Verification: rejected — it is ambiguous. '
                'Do not reuse that region.\n\n'
            ),
        }
        body = (
            f'Object in the previous reasoning: {ref}\n'
            f'Rejected coordinate: {coordinate}\n'
            + separated_reasons.get(reason, (
                'Verification: rejected — it cannot be verified reliably. '
                'Do not reuse that region.\n\n'
            ))
            + 'Recheck the image and localize the same object from scratch.\n'
            # The sole object-coordinate adjacency in this feedback context.
            + f'{ref}{DEFAULT_BOC_TOKEN}'
        )
    return body
