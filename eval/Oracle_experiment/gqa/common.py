"""Shared GQA CoT generation and final-answer utilities for oracle experiments."""

import torch

from constants import ALL_IMG_TOKENS_STR, COT_ACTIVATION, DEFAULT_GRD_TOKEN
from locals.datasets import SFT_DataCollator
from model.load_model import _build_inference_batch, _completion_metadata
from utils.eval_util import extract_all_box_str


# Mistral tokenization of ``[INST] What is your final answer? [/INST]`` used
# by the repository's existing ``GQADataset.cot_turn`` implementation.
GQA_FINAL_ANSWER_SUFFIX = [
    733, 16289, 28793, 1824, 349, 574, 1480, 4372, 28804,
    733, 28748, 16289, 28793,
]


def make_gqa_conversation(question):
    return [{
        'from': 'human',
        'value': ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION,
    }]


def normalize_gqa_answer(text):
    """Match the repository's existing GQA conversion normalization exactly."""
    return text.replace('</s>', '').rstrip('.').strip().lower()


def answer_is_correct(prediction, answer):
    return normalize_gqa_answer(prediction) == str(answer).strip().lower()


def generate_gqa_cot(model, preprocessor, image, conversation, max_new_tokens, temperature):
    """Generate one free CoT rollout and retain its full sequence IDs."""
    batch = _build_inference_batch(
        preprocessor, image, conversation=conversation
    )
    prompt_length = batch['input_ids'].shape[-1]
    response, _, sequences = model.condition_completion(
        batch,
        avoid_image_gen=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        record_bound_boxes=True,
    )
    return _completion_metadata(model, preprocessor.tokenizer, response, sequences, prompt_length), sequences


def generate_gqa_final_answer(model, preprocessor, image, thought_sequences,
                              prompt_length, max_new_tokens, temperature):
    """Ask for GQA's short final answer while retaining all CoT box bindings.

    This is the same token-level transition used by the repository's
    ``GQADataset.cot_turn``: every completed CoT coordinate is followed by an
    image token, and its parsed box is provided in ``box`` for REFbind.
    """
    tokenizer = preprocessor.tokenizer
    thought_ids = thought_sequences.squeeze()
    thought_text = tokenizer.batch_decode(
        thought_sequences[:, prompt_length:], skip_special_tokens=False
    )[0]
    thought_boxes = extract_all_box_str(thought_text, mistral=True)
    if any(box is None for box in thought_boxes):
        raise ValueError('CoT contains a malformed coordinate; cannot build GQA final-answer turn')

    eoc_indices = [-1] + torch.where(thought_ids == model.eoc_token_id)[0].tolist() + [
        thought_ids.shape[0] - 1
    ]
    input_id_chunks = []
    for index in range(len(eoc_indices) - 1):
        input_id_chunks.append(thought_ids[eoc_indices[index] + 1:eoc_indices[index + 1] + 1])
        if index < len(eoc_indices) - 2:
            if thought_ids[eoc_indices[index + 1] + 1].item() != model.input_img_id:
                input_id_chunks.append(torch.tensor([model.input_img_id], device=thought_ids.device))
    input_id_chunks.append(torch.tensor(GQA_FINAL_ANSWER_SUFFIX, device=thought_ids.device))
    final_input_ids = torch.cat(input_id_chunks)
    final_item = {
        'input_images': [image],
        'input_ids': final_input_ids,
        'box': [[0.0, 0.0, 1.0, 1.0]] + thought_boxes,
    }
    input_item = preprocessor(final_item)
    collator = SFT_DataCollator(tokenizer=tokenizer, sd_tokenizer=None)
    final_batch = collator([input_item])
    response, _, sequences = model.condition_completion(
        final_batch,
        avoid_image_gen=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        record_bound_boxes=True,
    )
    return {
        'response': response[0],
        'prediction': normalize_gqa_answer(response[0]),
        'thought': thought_text,
        'thought_boxes': thought_boxes,
        'sequences': sequences,
    }
