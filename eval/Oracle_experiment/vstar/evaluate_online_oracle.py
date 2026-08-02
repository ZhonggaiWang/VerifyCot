"""Evaluate online explicit-target GT grounding oracle on VStar.

For every model-generated ``<coor>``, only an unambiguous explicit reference
to a VStar-annotated target object is replaced with its ground-truth box.
Unmatched coordinates stay model-generated.  The decoder is never replayed
from a baseline trace, so each correction can change the subsequent CoT.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# This file lives in eval/Oracle_experiment/vstar/, three levels below the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from tqdm import tqdm

from constants import ALL_IMG_TOKENS_STR, COT_ACTIVATION, DEFAULT_GRD_TOKEN
from model.load_model import load_model, online_oracle_option_infer
from utils.coordinate_intervention import normalized_box_to_square_padding
from grounding_control.run_paths import (
    create_exact_output_layout,
    create_run_layout,
    write_run_config,
    write_run_status,
)


ORACLE_BOX_COORDINATE_SYSTEM = 'normalized_xyxy_on_center_padded_square'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument('--questions-path', required=True,
                        help='Official VStar test_questions.jsonl path.')
    parser.add_argument('--image-dir', required=True,
                        help='VStar root containing image paths from test_questions.jsonl.')
    parser.add_argument('--oracle-boxes-path', required=True,
                        help='JSONL produced by audit_boxes.py.')
    parser.add_argument(
        '--output', default=None,
        help='Exact results JSONL path; omit for the canonical run layout.',
    )
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--run-split', default='full_238')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--likelihood-reduction', choices=('mean', 'sum'), default='mean')
    parser.add_argument('--context-window-tokens', type=int, default=48,
                        help='Maximum normalized words before <coor> considered for matching.')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f'Invalid JSONL at {path}:{line_number}') from error
    return records


class VStarOnlineOracleDataset:
    """Canonical VStar options plus audited, normalized target-object boxes."""

    def __init__(self, questions_path, image_dir, oracle_boxes_path):
        self.questions = read_jsonl(questions_path)
        self.image_dir = Path(image_dir)
        oracle_records = read_jsonl(oracle_boxes_path)
        self.oracle_by_index = {record['sample_index']: record for record in oracle_records}
        if len(self.oracle_by_index) != len(oracle_records):
            raise ValueError('oracle-box JSONL has duplicate sample_index values')
        missing = [index for index in range(len(self.questions)) if index not in self.oracle_by_index]
        if missing:
            raise ValueError(f'oracle-box JSONL is missing {len(missing)} VStar samples, e.g. {missing[:5]}')

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        source_record = self.questions[index]
        oracle_record = self.oracle_by_index[index]
        image_path = self.image_dir / source_record['image']
        annotation_path = image_path.with_suffix('.json')
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        with annotation_path.open() as handle:
            annotation = json.load(handle)
        if annotation['question'] != oracle_record['question']:
            raise ValueError(f'Question mismatch for audited sample {index}')
        objects = oracle_record['target_objects']
        boxes = oracle_record['normalized_bboxes_xyxy']
        if len(objects) != len(boxes):
            raise ValueError(f'Target/box count mismatch for audited sample {index}')
        image_size = oracle_record.get('image_size')
        if not isinstance(image_size, dict):
            raise ValueError(f'Audited sample {index} has no image_size')
        image_width = image_size.get('width')
        image_height = image_size.get('height')
        square_padded_boxes = [
            normalized_box_to_square_padding(box, image_width, image_height)
            for box in boxes
        ]
        # ``aliases`` is deliberately just the annotated name.  The processor
        # normalizes articles, possessives, punctuation, and hyphenation; it
        # does not expand synonyms or drop discriminative modifiers.
        oracle_targets = [
            {'object': object_name, 'box': box, 'aliases': [object_name]}
            for object_name, box in zip(objects, square_padded_boxes)
        ]
        question = annotation['question']
        return {
            'sample_index': index,
            'question_id': source_record.get('question_id', str(index)),
            'category': source_record.get('category'),
            'image': source_record['image'],
            'image_path': image_path,
            'question': question,
            'options': annotation['options'],
            'label': 0,
            'conversation': [{
                'from': 'human',
                'value': ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION,
            }],
            'oracle_targets': oracle_targets,
            'source_oracle_boxes': boxes,
            'oracle_box_coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
            'source_image_size': {
                'width': image_width,
                'height': image_height,
            },
            'has_complete_question_target_coverage': oracle_record[
                'has_complete_question_target_coverage'
            ],
            'source_jsonl_label': source_record.get('label'),
        }


def load_existing_records(path):
    return [] if not Path(path).exists() else read_jsonl(path)


def accuracy(records, prediction_key):
    eligible = [record for record in records if record.get(prediction_key) is not None]
    if not eligible:
        return None, 0
    return (
        sum(record[prediction_key] == record['label'] for record in eligible) / len(eligible),
        len(eligible),
    )


def transition_counts(records):
    counts = Counter({
        'correct_to_wrong': 0,
        'wrong_to_correct': 0,
        'correct_to_correct': 0,
        'wrong_to_wrong': 0,
    })
    for record in records:
        baseline_correct = record['baseline_prediction'] == record['label']
        oracle_correct = record['oracle_prediction'] == record['label']
        if baseline_correct and not oracle_correct:
            counts['correct_to_wrong'] += 1
        elif not baseline_correct and oracle_correct:
            counts['wrong_to_correct'] += 1
        elif baseline_correct:
            counts['correct_to_correct'] += 1
        else:
            counts['wrong_to_wrong'] += 1
    return dict(counts)


def subset_summary(records):
    paired = [
        record for record in records
        if record.get('baseline_prediction') is not None and record.get('oracle_prediction') is not None
    ]
    baseline_accuracy, baseline_count = accuracy(paired, 'baseline_prediction')
    oracle_accuracy, oracle_count = accuracy(paired, 'oracle_prediction')
    forced_samples = [record for record in paired if record['intervention']['forced_coordinate_count'] > 0]
    return {
        'samples': len(paired),
        'baseline_accuracy': baseline_accuracy,
        'baseline_accuracy_count': baseline_count,
        'oracle_accuracy': oracle_accuracy,
        'oracle_accuracy_count': oracle_count,
        'oracle_minus_baseline': None if oracle_accuracy is None else oracle_accuracy - baseline_accuracy,
        'answer_changed_count': sum(
            record['baseline_prediction'] != record['oracle_prediction'] for record in paired
        ),
        'forced_sample_count': len(forced_samples),
        'no_forced_match_sample_count': len(paired) - len(forced_samples),
        'total_forced_coordinate_count': sum(
            record['intervention']['forced_coordinate_count'] for record in paired
        ),
        'correctness_transitions': transition_counts(paired),
    }


def make_summary(records, args):
    successful = [record for record in records if record.get('status') == 'ok']
    complete = [
        record for record in successful
        if record['has_complete_question_target_coverage']
    ]
    by_category = {}
    for category in sorted({record.get('category') for record in successful}):
        by_category[category] = subset_summary([
            record for record in successful if record.get('category') == category
        ])
    return {
        'total_records': len(records),
        'successful_records': len(successful),
        'errors': sum(record.get('status') == 'error' for record in records),
        'all_samples': subset_summary(successful),
        'complete_target_coverage_subset': subset_summary(complete),
        'by_category': by_category,
        'settings': {
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'likelihood_reduction': args.likelihood_reduction,
            'context_window_tokens': args.context_window_tokens,
            'oracle_mode': 'online_explicit_target_oracle',
            'coreference': 'off',
            'alias_policy': 'normalized full target-object phrase only',
            'oracle_box_coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
        },
    }


def main():
    args = parse_args()
    if args.output is None:
        layout = create_run_layout(
            dataset='vstar',
            split=args.run_split,
            study='oracle',
            method='always_gt',
            setting='default',
            run_id=args.run_id,
            output_root=args.output_root,
        )
    else:
        requested_output = Path(args.output)
        layout = create_exact_output_layout(
            dataset='vstar',
            split=args.run_split,
            study='oracle',
            method='always_gt',
            setting='default',
            run_id=args.run_id or requested_output.parent.name,
            output=requested_output,
        )
    layout.ensure_run_directories()
    output_path = layout.results_path
    write_run_config(layout, {
        'command': list(sys.argv),
        'arguments': vars(args),
        'inputs': {
            'questions': args.questions_path,
            'image_dir': args.image_dir,
            'oracle_boxes': args.oracle_boxes_path,
        },
        'components': {
            'generator': args.model_path,
            'verifier': 'oracle_target_matcher',
            'grounder': 'oracle_gt_box',
        },
        'coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
    })
    write_run_status(layout, 'running', completed_records=0)
    existing_records = [] if args.no_resume else load_existing_records(output_path)
    incompatible = [
        record for record in existing_records
        if record.get('oracle_box_coordinate_system') != ORACLE_BOX_COORDINATE_SYSTEM
    ]
    if incompatible:
        raise ValueError(
            'existing output uses the old or unknown oracle-box coordinate system; '
            'choose a new --output path or rerun with --no-resume'
        )
    completed_indices = {record['sample_index'] for record in existing_records if 'sample_index' in record}
    dataset = VStarOnlineOracleDataset(
        args.questions_path, args.image_dir, args.oracle_boxes_path
    )
    end_index = len(dataset) if args.max_samples is None else min(
        len(dataset), args.start_index + args.max_samples
    )
    indices = [index for index in range(args.start_index, end_index) if index not in completed_indices]
    print(f'VStar examples: {len(dataset)}; evaluating: {len(indices)}; resumed: {len(completed_indices)}')

    if indices:
        model, preprocessor = load_model(args.model_path, precision='fp16')
    output_mode = 'w' if args.no_resume else 'a'
    with output_path.open(output_mode) as handle:
        for index in tqdm(indices, desc='VStar online oracle evaluation'):
            item = dataset[index]
            record = {
                'sample_index': index,
                'question_id': item['question_id'],
                'image': item['image'],
                'category': item['category'],
                'question': item['question'],
                'options': item['options'],
                'label': item['label'],
                'source_jsonl_label': item['source_jsonl_label'],
                'oracle_targets': item['oracle_targets'],
                'source_oracle_boxes': item['source_oracle_boxes'],
                'oracle_box_coordinate_system': item['oracle_box_coordinate_system'],
                'source_image_size': item['source_image_size'],
                'has_complete_question_target_coverage': item[
                    'has_complete_question_target_coverage'
                ],
            }
            try:
                with Image.open(item['image_path']) as source_image:
                    image = source_image.convert('RGB')
                if image.size != (
                    item['source_image_size']['width'],
                    item['source_image_size']['height'],
                ):
                    raise ValueError(
                        f'image size {image.size} does not match audited size '
                        f'{item["source_image_size"]}'
                    )
                result = online_oracle_option_infer(
                    model, preprocessor, image, item['conversation'], item['options'],
                    oracle_targets=item['oracle_targets'],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    likelihood_reduction=args.likelihood_reduction,
                    further_instruct=True,
                    context_window_tokens=args.context_window_tokens,
                )
                record.update(result)
                record['status'] = 'ok'
            except Exception as error:
                record['status'] = 'error'
                record['error'] = f'{type(error).__name__}: {error}'
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()

    records = load_existing_records(output_path)
    summary = make_summary(records, args)
    summary_path = layout.summary_path
    with summary_path.open('w') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    error_records = summary.get('errors', 0)
    write_run_status(
        layout,
        'completed' if error_records == 0 else 'completed_with_errors',
        completed_records=summary.get('successful_records', 0),
        error_records=error_records,
        summary_path=str(summary_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Per-example results: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
