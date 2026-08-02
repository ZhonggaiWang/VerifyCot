"""Evaluate single-coordinate counterfactual CoT interventions on VStar.

The scoring procedure matches the repository's VStar command: generate a CoT,
append the option-selection instruction, then choose the option with minimum
token likelihood loss.  The counterfactual branch either replaces one generated
grounding box or suppresses its opening coordinate tag, then scores the freely
continued CoT.
"""

import argparse
import json
import secrets
import sys
from pathlib import Path

# This file lives in ``eval/Oracle_experiment/vstar/``.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from tqdm import tqdm

from constants import ALL_IMG_TOKENS_STR, COT_ACTIVATION, DEFAULT_GRD_TOKEN
from model.load_model import baseline_option_infer, counterfactual_option_infer, load_model
from grounding_control.run_paths import (
    create_exact_output_layout,
    create_run_layout,
    write_run_config,
    write_run_status,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument('--questions-path', required=True,
                        help='Official VStar test_questions.jsonl path.')
    parser.add_argument('--image-dir', required=True,
                        help='VStar repository root containing the relative image paths.')
    parser.add_argument('--annotation-root', default=None,
                        help='Root for paired VStar annotation JSON files (defaults to --image-dir).')
    parser.add_argument(
        '--output', default=None,
        help='Exact results JSONL path; omit for the canonical run layout.',
    )
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--run-split', default='full_238')
    parser.add_argument('--max-new-tokens', type=int, default=2048,
                        help='2048 matches the repository VStar benchmark command.')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--baseline-only', action='store_true',
                        help='Run the original CoT + option-likelihood baseline without intervention.')
    parser.add_argument('--likelihood-reduction', choices=('mean', 'sum'), default='mean')
    parser.add_argument('--perturb-mode', choices=('random_box', 'remove_grounding'),
                        default='random_box',
                        help='Replace one box with a random box, or suppress that grounding entirely.')
    parser.add_argument('--perturb-index', type=int, default=None,
                        help='Fixed 1-based coordinate index. Omit for per-sample random selection.')
    parser.add_argument('--perturb-position', choices=('random', 'first', 'last'), default='random',
                        help='Select a generated coordinate by position; overrides no explicit index.')
    parser.add_argument('--selection-seed', type=int, default=2026)
    parser.add_argument('--perturb-seed', type=int, default=2027)
    parser.add_argument('--random-seeds', action='store_true',
                        help='Generate and record fresh master seeds for this run.')
    parser.add_argument('--iou-min', type=float, default=0.0)
    parser.add_argument('--iou-max', type=float, default=0.1)
    parser.add_argument('--perturb-box-mode', choices=('random', 'same_shape'), default='random')
    parser.add_argument('--random-box-min-size', type=float, default=0.05)
    parser.add_argument('--random-box-max-size', type=float, default=0.5)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--paper-accuracy', type=float, default=None,
                        help='Optional published VStar accuracy as a fraction in [0, 1].')
    parser.add_argument('--paper-name', default='paper reference')
    parser.add_argument('--no-resume', action='store_true')
    return parser.parse_args()


class VStarJsonlDataset:
    """Use canonical paired annotations while retaining JSONL sample order.

    ``test_questions.jsonl`` contains shuffled A--D choices for standalone
    evaluation.  The V* benchmark's paired ``.json`` annotation instead has
    the canonical full-sentence options and places the correct option at index
    zero; that is the format expected by VoCoT's original VStarDataset.
    """

    def __init__(self, questions_path, image_dir, annotation_root=None):
        with open(questions_path) as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.image_dir = Path(image_dir)
        self.annotation_root = Path(annotation_root) if annotation_root else self.image_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.image_dir / record['image']
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        annotation_path = (self.annotation_root / record['image']).with_suffix('.json')
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        with annotation_path.open() as handle:
            annotation = json.load(handle)
        question = annotation['question']
        options = annotation['options']
        if not options:
            raise ValueError(f"VStar annotation has no options: {annotation_path}")
        conversation = [{
            'from': 'human',
            'value': ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION,
        }]
        return {
            'image': record['image'],
            'image_path': image_path,
            'question': question,
            'options': options,
            'label_index': 0,
            'label_letter': 'A (canonical option 0)',
            'source_jsonl_label': record['label'],
            'category': record.get('category'),
            'question_id': record.get('question_id', str(index)),
            'conversation': conversation,
        }


def load_records(path):
    if not path.exists():
        return []
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f'Invalid JSONL at {path}:{line_number}') from error
    return records


def accuracy(records, prediction_key):
    eligible = [record for record in records if record.get(prediction_key) is not None]
    if not eligible:
        return None, 0
    correct = sum(record[prediction_key] == record['label'] for record in eligible)
    return correct / len(eligible), len(eligible)


def make_summary(records, args):
    successful = [record for record in records if record.get('status') in ('ok', 'no_coordinate', 'baseline_only')]
    paired = [record for record in successful if record.get('counterfactual_prediction') is not None]
    baseline_accuracy, baseline_count = accuracy(successful, 'baseline_prediction')
    paired_baseline_accuracy, paired_count = accuracy(paired, 'baseline_prediction')
    counterfactual_accuracy, counterfactual_count = accuracy(paired, 'counterfactual_prediction')
    changed_answers = sum(
        record['baseline_prediction'] != record['counterfactual_prediction'] for record in paired
    )
    transitions = {
        'correct_to_wrong': 0,
        'wrong_to_correct': 0,
        'correct_to_correct': 0,
        'wrong_to_wrong': 0,
    }
    for record in paired:
        baseline_correct = record['baseline_prediction'] == record['label']
        counterfactual_correct = record['counterfactual_prediction'] == record['label']
        if baseline_correct and not counterfactual_correct:
            transitions['correct_to_wrong'] += 1
        elif not baseline_correct and counterfactual_correct:
            transitions['wrong_to_correct'] += 1
        elif baseline_correct and counterfactual_correct:
            transitions['correct_to_correct'] += 1
        else:
            transitions['wrong_to_wrong'] += 1
    summary = {
        'total_records': len(records),
        'successful_records': len(successful),
        'errors': sum(record.get('status') == 'error' for record in records),
        'no_coordinate': sum(record.get('status') == 'no_coordinate' for record in records),
        'intervened_records': len(paired),
        'baseline_accuracy_all': baseline_accuracy,
        'baseline_accuracy_all_count': baseline_count,
        'baseline_accuracy_paired_subset': paired_baseline_accuracy,
        'counterfactual_accuracy_paired_subset': counterfactual_accuracy,
        'paired_subset_count': paired_count,
        'counterfactual_accuracy_count': counterfactual_count,
        'answer_changed_rate': None if not paired else changed_answers / len(paired),
        'answer_changed_count': changed_answers,
        'correctness_transitions': transitions,
        'correctness_transition_rates': {
            name: None if not paired else count / len(paired)
            for name, count in transitions.items()
        },
        'settings': {
            'baseline_only': args.baseline_only,
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'likelihood_reduction': args.likelihood_reduction,
            'perturb_mode': args.perturb_mode,
            'perturb_index': args.perturb_index,
            'perturb_position': args.perturb_position,
            'selection_seed': args.selection_seed,
            'perturb_seed': args.perturb_seed,
            'iou_range': [args.iou_min, args.iou_max],
            'perturb_box_mode': args.perturb_box_mode,
            'random_box_size_range': [args.random_box_min_size, args.random_box_max_size],
        },
    }
    if baseline_accuracy is not None and counterfactual_accuracy is not None:
        summary['accuracy_drop_paired_subset'] = paired_baseline_accuracy - counterfactual_accuracy
    if args.paper_accuracy is not None and baseline_accuracy is not None:
        summary['paper_comparison'] = {
            'name': args.paper_name,
            'paper_accuracy': args.paper_accuracy,
            'baseline_minus_paper': baseline_accuracy - args.paper_accuracy,
        }
    return summary


def main():
    args = parse_args()
    if args.perturb_index is not None and args.perturb_position != 'random':
        raise ValueError('--perturb-index cannot be combined with --perturb-position')
    if args.random_seeds:
        args.selection_seed = secrets.randbits(63)
        args.perturb_seed = secrets.randbits(63)
        print(
            f'Fresh random seeds: selection={args.selection_seed}, '
            f'perturbation={args.perturb_seed}'
        )
    if args.perturb_position == 'first':
        resolved_perturb_index = 1
    elif args.perturb_position == 'last':
        resolved_perturb_index = 'last'
    else:
        resolved_perturb_index = args.perturb_index
    setting = (
        'index_{}'.format(args.perturb_index)
        if args.perturb_index is not None else args.perturb_position
    )
    study = 'baseline' if args.baseline_only else 'counterfactual'
    method = 'volcano_7b' if args.baseline_only else args.perturb_mode
    setting = 'default' if args.baseline_only else setting
    if args.output is None:
        layout = create_run_layout(
            dataset='vstar',
            split=args.run_split,
            study=study,
            method=method,
            setting=setting,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    else:
        requested_output = Path(args.output)
        layout = create_exact_output_layout(
            dataset='vstar',
            split=args.run_split,
            study=study,
            method=method,
            setting=setting,
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
            'annotation_root': args.annotation_root,
        },
        'components': {
            'generator': args.model_path,
            'intervention': None if args.baseline_only else args.perturb_mode,
        },
    })
    write_run_status(layout, 'running', completed_records=0)
    existing_records = [] if args.no_resume else load_records(output_path)
    completed_indices = {record['sample_index'] for record in existing_records if 'sample_index' in record}

    dataset = VStarJsonlDataset(args.questions_path, args.image_dir, args.annotation_root)
    end_index = len(dataset) if args.max_samples is None else min(
        len(dataset), args.start_index + args.max_samples
    )
    indices = [index for index in range(args.start_index, end_index) if index not in completed_indices]
    print(f'VStar examples: {len(dataset)}; evaluating: {len(indices)}; resumed: {len(completed_indices)}')

    if indices:
        model, preprocessor = load_model(args.model_path, precision='fp16')
    output_mode = 'w' if args.no_resume else 'a'
    with output_path.open(output_mode) as handle:
        for index in tqdm(indices, desc='VStar counterfactual evaluation'):
            item = dataset[index]
            label = item['label_index']
            # Per-example derivation avoids selecting the same coordinate rank
            # for every sample while allowing a complete run to be reproduced.
            selection_seed = None if args.selection_seed is None else args.selection_seed + 2 * index
            perturb_seed = None if args.perturb_seed is None else args.perturb_seed + 2 * index + 1
            record = {
                'sample_index': index,
                'question_id': item['question_id'],
                'image': item['image'],
                'category': item['category'],
                'question': item['question'],
                'options': item['options'],
                'label': label,
                'label_letter': item['label_letter'],
                'source_jsonl_label': item['source_jsonl_label'],
            }
            try:
                image = Image.open(item['image_path']).convert('RGB')
                if args.baseline_only:
                    result = baseline_option_infer(
                        model, preprocessor, image, item['conversation'], item['options'],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        likelihood_reduction=args.likelihood_reduction,
                        further_instruct=True,
                    )
                else:
                    result = counterfactual_option_infer(
                        model, preprocessor, image, item['conversation'], item['options'],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        perturb_index=resolved_perturb_index,
                        selection_seed=selection_seed,
                        perturb_seed=perturb_seed,
                        perturb_iou_range=(args.iou_min, args.iou_max),
                        perturb_mode=args.perturb_mode,
                        perturb_box_mode=args.perturb_box_mode,
                        random_box_min_size=args.random_box_min_size,
                        random_box_max_size=args.random_box_max_size,
                        likelihood_reduction=args.likelihood_reduction,
                        further_instruct=True,
                    )
                record.update(result)
                if args.baseline_only:
                    record['status'] = 'baseline_only'
                else:
                    record['status'] = 'ok' if result['counterfactual_prediction'] is not None else 'no_coordinate'
            except Exception as error:
                record['status'] = 'error'
                record['error'] = f'{type(error).__name__}: {error}'
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()

    records = load_records(output_path)
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
