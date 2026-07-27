"""Build an auditable GQA-val subset with exact target-instance GT boxes.

GQA validation questions contain visual pointers from question words to scene
graph object IDs.  This script joins those pointers with the official
``val_sceneGraphs.json`` and writes a fixed, stratified subset for grounding
intervention experiments.  It performs no model inference.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions-path', required=True,
                        help='Official GQA val_balanced_questions.json path.')
    parser.add_argument('--scene-graphs-path', required=True,
                        help='Official GQA val_sceneGraphs.json path.')
    parser.add_argument('--image-dir', required=True,
                        help='Directory containing {imageId}.jpg files.')
    parser.add_argument('--output', required=True, help='Output subset manifest JSONL.')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260724)
    parser.add_argument('--max-target-objects', type=int, default=3,
                        help='Keep questions with at most this many pointed scene-graph instances.')
    parser.add_argument('--stratify-by', choices=('none', 'structural_semantic'),
                        default='structural_semantic')
    return parser.parse_args()


def normalized_xyxy(object_record, image_width, image_height):
    x = float(object_record['x'])
    y = float(object_record['y'])
    width = float(object_record['w'])
    height = float(object_record['h'])
    if width <= 0 or height <= 0:
        return None
    x2, y2 = x + width, y + height
    if x < 0 or y < 0 or x2 > image_width or y2 > image_height:
        return None
    box = [x / image_width, y / image_height, x2 / image_width, y2 / image_height]
    if not all(math.isfinite(value) for value in box):
        return None
    return box


def semantic_object_ids(question_record, objects):
    """Extract scene-graph IDs explicitly present in GQA semantic arguments."""
    ids = []
    for step in question_record.get('semantic', []):
        argument = step.get('argument', '')
        if not isinstance(argument, str):
            continue
        for candidate in re.findall(r'\(([^()]+)\)', argument):
            if candidate in objects and candidate not in ids:
                ids.append(candidate)
    return ids


def build_candidate(question_id, question_record, scene_graph, image_dir, max_target_objects):
    annotations = question_record.get('annotations', {})
    question_pointers = annotations.get('question', {})
    if not isinstance(question_pointers, dict) or not question_pointers:
        return None, 'missing_question_visual_pointer'
    objects = scene_graph.get('objects', {})
    pointer_ids = []
    for object_id in question_pointers.values():
        if object_id not in pointer_ids:
            pointer_ids.append(object_id)
    if not all(object_id in objects for object_id in pointer_ids):
        return None, 'pointer_object_missing_from_scene_graph'
    program_ids = semantic_object_ids(question_record, objects)
    relevant_ids = list(pointer_ids)
    for object_id in program_ids:
        if object_id not in relevant_ids:
            relevant_ids.append(object_id)
    if not 1 <= len(relevant_ids) <= max_target_objects:
        return None, 'target_object_count_out_of_range'
    image_id = question_record.get('imageId')
    image_path = image_dir / f'{image_id}.jpg'
    if not image_path.is_file():
        return None, 'missing_image'
    image_width = scene_graph.get('width')
    image_height = scene_graph.get('height')
    if not isinstance(image_width, (int, float)) or not isinstance(image_height, (int, float)):
        return None, 'missing_scene_graph_image_size'
    if image_width <= 0 or image_height <= 0:
        return None, 'invalid_scene_graph_image_size'

    target_objects = []
    for object_id in relevant_ids:
        object_record = objects[object_id]
        normalized_box = normalized_xyxy(object_record, image_width, image_height)
        if normalized_box is None:
            return None, 'invalid_target_bbox'
        target_objects.append({
            'object_id': object_id,
            'name': object_record.get('name'),
            'pixel_bbox_xywh': [
                object_record['x'], object_record['y'], object_record['w'], object_record['h'],
            ],
            'normalized_bbox_xyxy': normalized_box,
            'question_word_positions': sorted(
                key for key, value in question_pointers.items() if value == object_id
            ),
            'target_sources': [
                source for source, present in (
                    ('question_visual_pointer', object_id in pointer_ids),
                    ('semantic_program', object_id in program_ids),
                ) if present
            ],
        })

    semantic_types = question_record.get('types', {})
    structural_type = semantic_types.get('structural', 'unknown')
    semantic_type = semantic_types.get('semantic', 'unknown')
    return {
        'question_id': question_id,
        'image_id': image_id,
        'image_path': str(image_path),
        'image_size': {'width': image_width, 'height': image_height},
        'question': question_record['question'],
        'answer': question_record['answer'],
        'full_answer': question_record.get('fullAnswer'),
        'types': semantic_types,
        'semantic': question_record.get('semantic', []),
        'semantic_str': question_record.get('semanticStr'),
        'question_target_object_ids': pointer_ids,
        'semantic_object_ids': program_ids,
        'relevant_object_ids': relevant_ids,
        'target_objects': target_objects,
        '_stratum': f'{structural_type}:{semantic_type}',
    }, None


def stratified_sample(candidates, count, seed, stratify_by):
    if count > len(candidates):
        raise ValueError(f'requested {count} samples but only {len(candidates)} are eligible')
    rng = random.Random(seed)
    if stratify_by == 'none':
        return rng.sample(candidates, count)
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate['_stratum']].append(candidate)
    for group in groups.values():
        rng.shuffle(group)
    total = len(candidates)
    allocations = {
        name: min(len(group), int(count * len(group) / total))
        for name, group in groups.items()
    }
    remaining = count - sum(allocations.values())
    # Largest-remainder allocation preserves the source type distribution while
    # producing exactly ``count`` rows.
    ranked_names = sorted(
        groups,
        key=lambda name: (
            count * len(groups[name]) / total - allocations[name],
            len(groups[name]),
            name,
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for name in ranked_names:
            if remaining == 0:
                break
            if allocations[name] < len(groups[name]):
                allocations[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError('could not allocate requested stratified sample')
    selected = []
    for name in sorted(groups):
        selected.extend(groups[name][:allocations[name]])
    rng.shuffle(selected)
    return selected


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError('--count must be positive')
    if args.max_target_objects <= 0:
        raise ValueError('--max-target-objects must be positive')
    questions_path = Path(args.questions_path)
    scene_graphs_path = Path(args.scene_graphs_path)
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)
    summary_path = output_path.with_suffix('.summary.json')

    with questions_path.open() as handle:
        questions = json.load(handle)
    with scene_graphs_path.open() as handle:
        scene_graphs = json.load(handle)

    candidates = []
    rejected = Counter()
    for question_id, question_record in questions.items():
        image_id = question_record.get('imageId')
        scene_graph = scene_graphs.get(image_id)
        if scene_graph is None:
            rejected['missing_scene_graph'] += 1
            continue
        candidate, reason = build_candidate(
            question_id, question_record, scene_graph, image_dir, args.max_target_objects
        )
        if reason is not None:
            rejected[reason] += 1
            continue
        candidates.append(candidate)

    selected = stratified_sample(candidates, args.count, args.seed, args.stratify_by)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as handle:
        for sample_index, candidate in enumerate(selected):
            candidate = dict(candidate)
            candidate['sample_index'] = sample_index
            candidate.pop('_stratum')
            handle.write(json.dumps(candidate, ensure_ascii=False) + '\n')

    selected_type_counts = Counter(
        f"{candidate['types'].get('structural', 'unknown')}:"
        f"{candidate['types'].get('semantic', 'unknown')}"
        for candidate in selected
    )
    summary = {
        'source_questions_path': str(questions_path),
        'source_scene_graphs_path': str(scene_graphs_path),
        'image_dir': str(image_dir),
        'total_source_questions': len(questions),
        'eligible_candidates': len(candidates),
        'selected_samples': len(selected),
        'selection_seed': args.seed,
        'max_target_objects': args.max_target_objects,
        'stratify_by': args.stratify_by,
        'eligible_type_counts': dict(sorted(Counter(
            candidate['_stratum'] for candidate in candidates
        ).items())),
        'selected_type_counts': dict(sorted(selected_type_counts.items())),
        'rejected_counts': dict(sorted(rejected.items())),
        'target_definition': (
            'Ordered union of object IDs explicitly pointed to by question annotations '
            'and object IDs explicitly named in the semantic program.'
        ),
        'coordinate_convention': {
            'input_bbox': '[x, y, width, height] in GQA scene-graph image pixels',
            'output_bbox': '[x1, y1, x2, y2] normalized to [0, 1]',
        },
    }
    with summary_path.open('w') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(
        f'GQA candidates: {len(candidates)} / {len(questions)}; '
        f'selected: {len(selected)}\n'
        f'Manifest: {output_path}\n'
        f'Summary: {summary_path}'
    )


if __name__ == '__main__':
    main()
