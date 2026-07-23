from genericpath import samestat
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, Optional
from dataclasses import dataclass, field
from locals.datasets import SFT_DataCollator, WrappedDataset
from lightning.pytorch import seed_everything
from torchvision import transforms
from constants import *
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from locals.datasets.preprocessor import VoCoT_InputProcessor
from omegaconf import OmegaConf
from utils.util import instantiate_from_config
from model.language_model.volcano_llama import VolCanoLlamaForCausalLM,VolCanoConfig
from model.language_model.volcano_mistral import VolCanoMistralForCausalLM, VolCanoMistralConfig
from transformers import AutoTokenizer, LlamaTokenizer, LogitsProcessor, LogitsProcessorList
import transformers
from peft import PeftConfig, PeftModel
from argparse import ArgumentParser
import os
import torch.distributed as dist
from utils.logger import setup_logger
import json
import random
import tqdm

def rank0_print(args, res):
    if args.local_rank==0 or args.local_rank == -1:
        print(res)

def get_output_name(args, mid_output=True):
    if mid_output:
        return os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, args.local_rank))
    else:
        return os.path.join(args.output_dir, 
                            '{}.json'.format(args.dataset_name))

def get_all_output_names(args):
    return [os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, r)) for r in range(args.n_gpus)]

class CLIPTransform:
    def __init__(self, transform, square_size=None):
        self.transform = transform
        self.square_size = square_size
        self.image_mean = transform.image_mean
    
    def __call__(self, image):
        if self.square_size is not None:
            image = image.resize((self.square_size, self.square_size))
        try:
            tmp = torch.tensor(self.transform(image)['pixel_values'][0])
        except:
            tmp = torch.tensor(self.transform(Image.new(image.mode, (32, 32), (0,0,0)))['pixel_values'][0])
        return tmp


class RandomCoordinateLogitsProcessor(LogitsProcessor):
    """Replace each generated coordinate span with a dynamically sampled box.

    The processor waits for the model to emit ``<coor>`` itself.  It then forces
    the coordinate text and closing ``</coor>`` token.  Consequently the normal
    ``generate_box`` path parses exactly the text seen in the output and binds
    the matching visual region without any special handling in the model.
    """

    def __init__(self, tokenizer, prompt_length: int, seed: Optional[int] = None,
                 min_box_size: float = 0.05, precision: int = 3,
                 max_randomized_coors: Optional[int] = None):
        if not 0 < min_box_size <= 1:
            raise ValueError("min_box_size must be in (0, 1]")
        if max_randomized_coors is not None and max_randomized_coors < 0:
            raise ValueError("max_randomized_coors must be non-negative or None")
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.min_box_size = min_box_size
        self.precision = precision
        self.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
        self.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
        self.rng = random.Random(seed)
        self.max_randomized_coors = max_randomized_coors
        self.sampled_boxes = []
        self._target_suffixes = []

    def _sample_box(self):
        x_min = self.rng.uniform(0.0, 1.0 - self.min_box_size)
        y_min = self.rng.uniform(0.0, 1.0 - self.min_box_size)
        x_max = self.rng.uniform(x_min + self.min_box_size, 1.0)
        y_max = self.rng.uniform(y_min + self.min_box_size, 1.0)
        return tuple(round(value, self.precision) for value in (x_min, y_min, x_max, y_max))

    def _new_target_suffix(self):
        box = self._sample_box()
        box_text = ",".join(f"{value:.{self.precision}f}" for value in box)
        # Decoding the special <coor> token already inserts its trailing space.
        # Adding another leading space here would break extract_box_str's format.
        suffix_text = f"{box_text}{DEFAULT_EOC_TOKEN}"
        suffix_ids = self.tokenizer(suffix_text, add_special_tokens=False).input_ids
        if not suffix_ids or suffix_ids[-1] != self.eoc_token_id:
            raise ValueError(f"Could not tokenize a coordinate suffix: {suffix_text}")
        self.sampled_boxes.append(box)
        self._target_suffixes.append(suffix_ids)
        return suffix_ids

    @staticmethod
    def _force_token(scores, token_id: int):
        scores.fill_(float("-inf"))
        scores[:, token_id] = 0
        return scores

    def __call__(self, input_ids, scores):
        # The existing VoCoT generation/binding path is batch-size one.
        if input_ids.shape[0] != 1:
            raise ValueError("RandomCoordinateLogitsProcessor supports batch size 1 only")

        generated_ids = input_ids[0, self.prompt_length:].tolist()
        boc_positions = [idx for idx, token_id in enumerate(generated_ids) if token_id == self.boc_token_id]
        eoc_positions = [idx for idx, token_id in enumerate(generated_ids) if token_id == self.eoc_token_id]
        if not boc_positions or (eoc_positions and eoc_positions[-1] > boc_positions[-1]):
            return scores

        # A coordinate is active: its index equals the number of completed
        # coordinate spans.  Sample only once, immediately after <coor> appears.
        coordinate_index = len(eoc_positions)
        if (self.max_randomized_coors is not None
                and coordinate_index >= self.max_randomized_coors):
            return scores
        if coordinate_index == len(self._target_suffixes):
            target_suffix = self._new_target_suffix()
        else:
            target_suffix = self._target_suffixes[coordinate_index]

        current_boc = boc_positions[-1]
        observed_suffix = generated_ids[current_boc + 1:]
        expected_prefix = target_suffix[:len(observed_suffix)]
        if observed_suffix != expected_prefix:
            raise RuntimeError("Generated coordinate tokens diverged from the forced coordinate suffix")
        if len(observed_suffix) >= len(target_suffix):
            return scores
        return self._force_token(scores, target_suffix[len(observed_suffix)])



def load_model(model_path, device='cuda:0', precision='bf16'):
    config_class = VolCanoMistralConfig
    model_class = VolCanoMistralForCausalLM
    tokenizer_class = AutoTokenizer
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        use_fast=True,
        trust_remote_code=True
    )
    
    llama_config = config_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, config=llama_config)

    model.input_img_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMG_TOKEN)
    model.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
    model.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
    model.tokenizer = tokenizer
    model.sub_image_bind = False

    if precision == 'bf16':
        model.to(torch.bfloat16)
    elif precision == 'fp16':
        model.to(torch.float16)
    elif precision == 'fp32':
        pass
    else:
        raise ValueError('precision must be fp16, bf16, or fp32')
    model.eval()
    model.to(device)

    resize2square = False
    output_vis_processor = transforms.Compose(
                [
                    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(1024),
                    # transforms.RandomHorizontalFlip(), # comment here
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
    input_vis_processor = transforms.Compose(
            [
                transforms.Resize((448, 448) if resize2square else 448, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(448),
                # transforms.RandomHorizontalFlip(), comment here
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
        )
    if hasattr(model.vision_encoder, 'image_processor'):
        input_vis_processor = model.vision_encoder.image_processor
        if resize2square:
            tmp_size = input_vis_processor.size['shortest_edge']
        else:
            tmp_size = None
        input_vis_processor = CLIPTransform(input_vis_processor, square_size=tmp_size)
    # tokenizer = LlamaTokenizer.from_pretrained('eval/debug/edit_gpt_emu_tokenizer')

    model.image_processor = None
    preprocessor = VoCoT_InputProcessor(tokenizer=tokenizer, input_image_processor = input_vis_processor, use_mistral=True,
                                                output_image_processor= output_vis_processor, merge_in_out_image=True, expand2square=True, inference = True)

    return model, preprocessor

def infer(model, preprocessor, image, query, cot=True, max_new_tokens=1024, temperature=0.0,
          randomize_coor=False, random_coor_seed=None, random_coor_min_size=0.05,
          max_randomized_coors=None, return_metadata=False):
    if cot:
        query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + query + COT_ACTIVATION
    else:
        query = ALL_IMG_TOKENS_STR + '\n' + query
    conv = [{'from': 'human', 'value':query}]
    item = {'input_images': [image], 'conversation': conv}
    input_item = preprocessor(item)
    data_collator = SFT_DataCollator(tokenizer=preprocessor.tokenizer, sd_tokenizer=None)
    batch = data_collator([input_item])
    coor_processor = None
    if randomize_coor:
        coor_processor = RandomCoordinateLogitsProcessor(
            preprocessor.tokenizer,
            prompt_length=batch['input_ids'].shape[-1],
            seed=random_coor_seed,
            min_box_size=random_coor_min_size,
            max_randomized_coors=max_randomized_coors,
        )
    txt_res, out_imgs, txt_ids = model.condition_completion(batch, avoid_image_gen=True, 
                                                            max_new_tokens=max_new_tokens, temperature=temperature,
                                                            logits_processor=(LogitsProcessorList([coor_processor])
                                                                              if coor_processor is not None else None))

    if return_metadata:
        return txt_res, {'forced_boxes': [] if coor_processor is None else coor_processor.sampled_boxes}
    return txt_res
            

if __name__=='__main__':
    from PIL import Image
    tmp_image = Image.open('eval/debug/tmp.jpg')
    model_path = '/mnt/bn/yangmin-priv/luoruipu/checkpoints/LLaVA-clip336px-obj-represent-Mistral-1e-5-3072-instruct_llava+shikraCoT75per+GPTQTA+lvis-cot/'
    model, preprocessor = load_model(model_path,precision='fp16')
    res1 = infer(model, preprocessor, tmp_image, 'Is there a event "the cat is below the bed" in this image?', cot=True)
    res = infer(model, preprocessor, tmp_image, 'Why is the cat on the bed?', cot=True)
    res_no_cot = infer(model, preprocessor, tmp_image, 'Describe the image.', cot=True)
    print(res1)
    print(res)
    print(res_no_cot)
