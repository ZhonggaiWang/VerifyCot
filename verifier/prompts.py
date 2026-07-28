"""Temporary repair prompts.  These strings are never committed to CoT."""

from typing import Literal, Sequence

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN


RepairMode = Literal[
    'blind_retry', 'binary_feedback', 'typed_feedback', 'concise_typed_feedback',
    'separated_reference_feedback', 'separated_reference_feedback_v2'
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
        'separated_reference_feedback', 'separated_reference_feedback_v2'
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
    elif repair_mode == 'separated_reference_feedback':
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
    else:
        # V2 keeps historical object/q text separated and additionally makes
        # the spatial relation between q and r explicit.  The original
        # separated mode above stays available as the recorded baseline.
        separated_v2_reasons = {
            'wrong_object': (
                'Verification: rejected — the coordinate localizes a different object, '
                'not the object described above.\n\n'
            ),
            'partial_coverage': (
                'Verification: rejected — the coordinate covers only part of the '
                'object described above.\n\n'
            ),
            'ambiguous': (
                'Verification: rejected — the coordinate contains multiple possible '
                'objects and does not uniquely identify the object described above.\n\n'
            ),
        }
        body = (
            f'Object in the previous reasoning: {ref}\n'
            f'Rejected coordinate: {coordinate}\n'
            + separated_v2_reasons.get(reason, (
                'Verification: rejected — the coordinate cannot be verified for the '
                'object described above.\n\n'
            ))
            + 'The replacement must localize the same object in a spatially different '
              'image region. Do not copy the rejected coordinate or reuse its immediate '
              'surrounding region.\n\n'
            + 'Inspect the image again, identify the same object, and output its replacement '
              'coordinate.\n'
            # The sole object-coordinate adjacency in this feedback context.
            + f'{ref}{DEFAULT_BOC_TOKEN}'
        )
    return body


def build_repair_prompt_text_only_q(
        object_reference: str, rejected_bbox: Sequence[float], reason: str,
        repair_mode: RepairMode = 'separated_reference_feedback') -> str:
    """Text-only-q counterpart for every repair prompt mode.

    ``q`` is rendered as ``[x1,y1,x2,y2]`` rather than a completed coordinate
    span.  Therefore this changes no model setting and leaves H_t/refinement-r
    binding intact, while preventing only temporary q from entering REFbind.
    """
    if repair_mode == 'blind_retry':
        return ''
    if repair_mode not in {
        'binary_feedback', 'typed_feedback', 'concise_typed_feedback',
        'separated_reference_feedback', 'separated_reference_feedback_v2'
    }:
        raise ValueError(f'unknown repair_mode: {repair_mode}')
    ref = object_reference.strip() or 'the current object reference'
    numeric_coordinate = f'[{format_box(rejected_bbox)}]'

    if repair_mode == 'binary_feedback':
        return (
            f'Previous reasoning object: {ref}\n'
            f'Rejected coordinate (text only): {numeric_coordinate}\n'
            'This coordinate was rejected because it does not reliably localize the '
            'referenced object.\n\n'
            f'I will keep "{ref}" unchanged, re-reason about the same object, and '
            'select a new coordinate without reusing the rejected region.\n\n'
            f'I re-reason as {ref}{DEFAULT_BOC_TOKEN}'
        )

    if repair_mode == 'typed_feedback':
        typed_reasons = {
            'wrong_object': (
                'This coordinate was rejected because it localizes a different object.\n\n'
                f'I will keep "{ref}" unchanged and re-reason about the same object. '
                'I will relocate it to a different region, avoiding reuse of the rejected '
                'region and preferably using low overlap with it.\n\n'
            ),
            'partial_coverage': (
                'This coordinate was rejected because it covers only part of the object.\n\n'
                f'I will keep "{ref}" unchanged and re-reason about the same object. '
                'I will adjust or expand the region to cover the complete object while '
                'excluding unnecessary unrelated regions.\n\n'
            ),
            'ambiguous': (
                'This coordinate was rejected because it is ambiguous and contains '
                'multiple possible objects.\n\n'
                f'I will keep "{ref}" unchanged and re-reason about the same object. '
                'I will select a tighter, more discriminative region that uniquely '
                'localizes it.\n\n'
            ),
        }
        return (
            f'Previous reasoning object: {ref}\n'
            f'Rejected coordinate (text only): {numeric_coordinate}\n'
            + typed_reasons.get(reason, (
                'This coordinate could not be verified reliably.\n\n'
                f'I will keep "{ref}" unchanged and re-reason about the same object. '
                'I will generate a clearer region that uniquely localizes it.\n\n'
            ))
            + f'I re-reason as {ref}{DEFAULT_BOC_TOKEN}'
        )

    if repair_mode == 'concise_typed_feedback':
        concise_reasons = {
            'wrong_object': (
                'Verification: rejected — this region localizes a different object. '
                'Do not reuse this region.\n\n'
            ),
            'partial_coverage': (
                'Verification: rejected — this region covers only part of the object. '
                'Do not reuse this region unchanged.\n\n'
            ),
            'ambiguous': (
                'Verification: rejected — this region is ambiguous. '
                'Do not reuse this region.\n\n'
            ),
        }
        return (
            f'Previous reasoning object: {ref}\n'
            f'Rejected coordinate (text only): {numeric_coordinate}\n'
            + concise_reasons.get(reason, (
                'Verification: rejected — this region cannot be verified reliably. '
                'Do not reuse this region.\n\n'
            ))
            + 'Recheck the image and localize the same object from scratch.\n'
            + f'{ref}{DEFAULT_BOC_TOKEN}'
        )

    if repair_mode == 'separated_reference_feedback':
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
        return (
            f'Object in the previous reasoning: {ref}\n'
            f'Rejected coordinate (text only): {numeric_coordinate}\n'
            + separated_reasons.get(reason, (
            'Verification: rejected — it cannot be verified reliably. '
            'Do not reuse that region.\n\n'
        ))
        + 'Recheck the image and localize the same object from scratch.\n'
        # This is the only object-coordinate adjacency in the sandbox.
        + f'{ref}{DEFAULT_BOC_TOKEN}'
        )

    separated_v2_reasons = {
        'wrong_object': (
            'Verification: rejected — the coordinate localizes a different object, '
            'not the object described above.\n\n'
        ),
        'partial_coverage': (
            'Verification: rejected — the coordinate covers only part of the '
            'object described above.\n\n'
        ),
        'ambiguous': (
            'Verification: rejected — the coordinate contains multiple possible '
            'objects and does not uniquely identify the object described above.\n\n'
        ),
    }
    return (
        f'Object in the previous reasoning: {ref}\n'
        f'Rejected coordinate (text only): {numeric_coordinate}\n'
        + separated_v2_reasons.get(reason, (
            'Verification: rejected — the coordinate cannot be verified for the '
            'object described above.\n\n'
        ))
        + 'The replacement must localize the same object in a spatially different '
          'image region. Do not copy the rejected coordinate or reuse its immediate '
          'surrounding region.\n\n'
        + 'Inspect the image again, identify the same object, and output its replacement '
          'coordinate.\n'
        + f'{ref}{DEFAULT_BOC_TOKEN}'
    )
