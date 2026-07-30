"""Build a controlled five-way GQA object-coordinate verifier benchmark.

The benchmark intentionally uses only a canonical object reference and an
image with one uniformly rendered candidate box as the model-facing input.
Full GT and construction metadata remain in JSONL solely for auditing and
evaluation.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.Oracle_experiment.generate_datasets.verifier_benchmark import (
    generate_aligned,
    generate_ambiguous,
    generate_partial_coverage,
    generate_unsupported,
    generate_wrong_object,
)
from eval.Oracle_experiment.generate_datasets.verifier_benchmark.common import scene_objects
from eval.Oracle_experiment.generate_datasets.verifier_benchmark.render import (
    render_candidate,
)


GENERATORS = {
    'aligned': generate_aligned,
    'wrong_object': generate_wrong_object,
    'partial_coverage': generate_partial_coverage,
    'ambiguous': generate_ambiguous,
    'unsupported': generate_unsupported,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest-path',
        default='output/gqa/annotations/oracle_val_1000/manifest.jsonl',
        help='GQA manifest containing explicit target object IDs.',
    )
    parser.add_argument(
        '--scene-graphs-path',
        default='/data/zhonggai/GQA/val_sceneGraphs.json',
        help='Official GQA scene graph JSON used to obtain every object in each image.',
    )
    parser.add_argument(
        '--image-dir',
        default='/data/zhonggai/GQA/images',
        help='Image directory; {image_id}.jpg takes precedence over manifest absolute paths.',
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--count-per-class', type=int, default=300)
    for label in GENERATORS:
        parser.add_argument(
            f'--{label.replace("_", "-")}-count',
            type=int,
            default=None,
            help=f'Override --count-per-class for {label}.',
        )
    parser.add_argument('--seed', type=int, default=20260729)
    parser.add_argument('--dev-fraction', type=float, default=0.20)
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument(
        '--allow-shortfall',
        action='store_true',
        help='Write all constructible records instead of failing when a class is short.',
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_split(image_id: str, seed: int, dev_fraction: float) -> str:
    digest = hashlib.sha256(f'{seed}:{image_id}'.encode('utf-8')).digest()
    value = int.from_bytes(digest[:8], 'big') / float(2 ** 64)
    return 'dev' if value < dev_fraction else 'test'


def collect_unique_targets(
        manifest: Iterable[Dict[str, object]],
        scenes: Dict[str, object],
        image_dir: Path) -> List[Dict[str, object]]:
    """Deduplicate manifest targets by image/object while retaining provenance."""
    grouped = {}
    for sample in manifest:
        image_id = str(sample['image_id'])
        scene = scenes.get(image_id)
        if not isinstance(scene, dict):
            continue
        width, height = int(scene['width']), int(scene['height'])
        converted_objects = scene_objects(scene)
        by_id = {item['object_id']: item for item in converted_objects}
        image_path = image_dir / f'{image_id}.jpg'
        if not image_path.is_file():
            fallback = Path(str(sample.get('image_path', '')))
            image_path = fallback if fallback.is_file() else image_path
        if not image_path.is_file():
            continue
        for manifest_target in sample.get('target_objects', []):
            object_id = str(manifest_target.get('object_id'))
            target = by_id.get(object_id)
            if target is None:
                continue
            key = (image_id, object_id)
            if key not in grouped:
                grouped[key] = {
                    'image_id': image_id,
                    'image_path': str(image_path),
                    'image_width': width,
                    'image_height': height,
                    'target': target,
                    'objects': converted_objects,
                    'source_question_ids': [],
                }
            question_id = str(sample.get('question_id'))
            if question_id not in grouped[key]['source_question_ids']:
                grouped[key]['source_question_ids'].append(question_id)
    return list(grouped.values())


def call_generator(
        label: str,
        source: Dict[str, object],
        rng: random.Random) -> Optional[Dict[str, object]]:
    common = {
        'target': source['target'],
        'image_width': source['image_width'],
        'image_height': source['image_height'],
        'rng': rng,
    }
    if label in ('wrong_object', 'ambiguous', 'unsupported'):
        common['objects'] = source['objects']
    return GENERATORS[label](**common)


def generate_class(
        label: str,
        sources: List[Dict[str, object]],
        count: int,
        rng: random.Random) -> List[Dict[str, object]]:
    order = list(sources)
    rng.shuffle(order)
    records = []
    for source in order:
        generated = call_generator(label, source, rng)
        if generated is None:
            continue
        generated.update({
            'image_id': source['image_id'],
            'source_image': source['image_path'],
            'image_size': {
                'width': source['image_width'],
                'height': source['image_height'],
            },
            'source_question_ids': source['source_question_ids'],
        })
        records.append(generated)
        if len(records) == count:
            break
    return records


def main():
    args = parse_args()
    if args.count_per_class < 0:
        raise ValueError('--count-per-class cannot be negative')
    if not 0.0 <= args.dev_fraction < 1.0:
        raise ValueError('--dev-fraction must be in [0, 1)')
    output_dir = Path(args.output_dir)
    manifest_output = output_dir / 'benchmark.jsonl'
    summary_output = output_dir / 'benchmark.summary.json'
    if manifest_output.exists() or summary_output.exists():
        raise FileExistsError(
            f'benchmark output already exists under {output_dir}; choose a new --output-dir'
        )

    manifest = read_jsonl(Path(args.manifest_path))
    with Path(args.scene_graphs_path).open(encoding='utf-8') as handle:
        scenes = json.load(handle)
    sources = collect_unique_targets(manifest, scenes, Path(args.image_dir))
    if not sources:
        raise RuntimeError('no valid unique target objects were joined to GQA scene graphs')

    requested_counts = {
        label: (
            getattr(args, f'{label}_count')
            if getattr(args, f'{label}_count') is not None
            else args.count_per_class
        )
        for label in GENERATORS
    }
    if any(count < 0 for count in requested_counts.values()):
        raise ValueError('per-class counts cannot be negative')

    records = []
    generated_counts = {}
    for label_index, (label, requested_count) in enumerate(requested_counts.items()):
        class_rng = random.Random(args.seed + 1009 * label_index)
        class_records = generate_class(label, sources, requested_count, class_rng)
        generated_counts[label] = len(class_records)
        if len(class_records) < requested_count and not args.allow_shortfall:
            raise RuntimeError(
                f'{label}: requested {requested_count}, constructed {len(class_records)}; '
                'use --allow-shortfall to retain the available records'
            )
        records.extend(class_records)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class_handles = {
        label: (output_dir / f'{label}.jsonl').open('w', encoding='utf-8')
        for label in GENERATORS
    }
    split_counts = Counter()
    try:
        with manifest_output.open('w', encoding='utf-8') as combined:
            for sample_index, record in enumerate(records):
                label = record['reason']
                split = stable_split(record['image_id'], args.seed, args.dev_fraction)
                event_id = (
                    f"gqa:{record['image_id']}:{record['target_object_id']}:"
                    f"{label}:{sample_index}"
                )
                rendered_relpath = Path('renders') / split / label / f'{sample_index:06d}.png'
                record.update({
                    'sample_index': sample_index,
                    'event_id': event_id,
                    'split': split,
                    'rendered_image': (
                        None if args.no_render else str(output_dir / rendered_relpath)
                    ),
                    'model_input': {
                        'object_reference': record['object_reference'],
                        'image': None if args.no_render else str(output_dir / rendered_relpath),
                    },
                })
                if not args.no_render:
                    render_candidate(
                        Path(record['source_image']),
                        record['candidate_box_pixel_xyxy'],
                        output_dir / rendered_relpath,
                    )
                serialized = json.dumps(record, ensure_ascii=False)
                combined.write(serialized + '\n')
                per_class_handles[label].write(serialized + '\n')
                split_counts[(split, label)] += 1
    finally:
        for handle in per_class_handles.values():
            handle.close()

    summary = {
        'builder': str(Path(__file__).resolve()),
        'source_manifest': str(Path(args.manifest_path)),
        'source_scene_graphs': str(Path(args.scene_graphs_path)),
        'source_image_dir': str(Path(args.image_dir)),
        'seed': args.seed,
        'dev_fraction': args.dev_fraction,
        'unique_target_count': len(sources),
        'requested_counts': requested_counts,
        'generated_counts': generated_counts,
        'total_records': len(records),
        'split_class_counts': {
            split: {
                label: split_counts[(split, label)]
                for label in GENERATORS
            }
            for split in ('dev', 'test')
        },
        'model_visible_fields': [
            'model_input.object_reference',
            'model_input.image',
        ],
        'label_policy': {
            'aligned': 'slightly jittered GT box with IoU>=0.70 and target coverage>=0.85',
            'wrong_object': 'GT box of a differently named object with low target overlap',
            'partial_coverage': '25%-50% area crop contained inside target GT box',
            'ambiguous': 'one union box containing target and a different nearby object',
            'unsupported': (
                'random region avoiding retained non-global GQA scene-graph objects; '
                'requires manual spot audit because scene graphs may be incomplete'
            ),
        },
        'rendering': {
            'candidate_outline_rgb': [255, 0, 255],
            'label_text_drawn': False,
            'gt_visible_to_model': False,
        },
    }
    with summary_output.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f'Built {len(records)} verifier records in {output_dir}')
    print(json.dumps(summary['generated_counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
