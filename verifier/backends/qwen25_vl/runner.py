"""Lazy local Qwen2.5-VL runner.

The imports are intentionally delayed so the original VoCoT environment can
import verifier modules without requiring a newer Transformers installation.
"""

from typing import Any, Mapping, Optional, Protocol, Sequence

from .option_likelihood import SingleTokenOptionScores


DEFAULT_MIN_PIXELS = 4 * 28 * 28
DEFAULT_MAX_PIXELS = 512 * 28 * 28


class Qwen25VLRunner(Protocol):
    def generate(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Generate one verifier response for already constructed messages."""


class LocalQwen25VLRunner:
    """Load and run Qwen2.5-VL in a Qwen-compatible Python environment."""

    def __init__(
            self,
            model_path: str,
            device: str = 'cuda:0',
            dtype: str = 'bfloat16',
            max_new_tokens: int = 64,
            min_pixels: int = DEFAULT_MIN_PIXELS,
            max_pixels: int = DEFAULT_MAX_PIXELS,
            local_files_only: bool = True,
            attn_implementation: str = 'sdpa'):
        if not model_path:
            raise ValueError('model_path is required')
        if max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        if (
            not isinstance(min_pixels, int)
            or isinstance(min_pixels, bool)
            or min_pixels <= 0
        ):
            raise ValueError('min_pixels must be a positive integer')
        if (
            not isinstance(max_pixels, int)
            or isinstance(max_pixels, bool)
            or max_pixels <= 0
        ):
            raise ValueError('max_pixels must be a positive integer')
        if min_pixels > max_pixels:
            raise ValueError('min_pixels must not exceed max_pixels')
        self.model_path = str(model_path)
        self.device = str(device)
        self.dtype = str(dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.local_files_only = bool(local_files_only)
        self.attn_implementation = str(attn_implementation)
        self._model = None
        self._processor = None
        self._input_device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError(
                'LocalQwen25VLRunner requires a Qwen2.5-VL-compatible '
                'Transformers environment (for example Transformers 4.49). '
                'Do not upgrade the original VoCoT generator environment.'
            ) from error

        dtype_by_name = {
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float16': torch.float16,
            'fp16': torch.float16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        if self.dtype == 'auto':
            torch_dtype = 'auto'
        else:
            try:
                torch_dtype = dtype_by_name[self.dtype.lower()]
            except KeyError as error:
                raise ValueError(
                    f'unsupported Qwen dtype: {self.dtype!r}'
                ) from error

        load_kwargs = {
            'torch_dtype': torch_dtype,
            'local_files_only': self.local_files_only,
            'attn_implementation': self.attn_implementation,
        }
        if self.device == 'auto' or self.device.startswith('cuda'):
            load_kwargs['device_map'] = self.device
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        if self.device != 'auto' and not self.device.startswith('cuda'):
            self._model.to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            use_fast=False,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self._input_device = next(self._model.parameters()).device

    def generate(self, messages: Sequence[Mapping[str, Any]]) -> str:
        self._load()
        inputs = self._processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        )
        inputs = inputs.to(self._input_device)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def score_single_token_options(
            self,
            messages: Sequence[Mapping[str, Any]],
            options: Mapping[str, str],
    ) -> SingleTokenOptionScores:
        """Score named one-token completions with one multimodal forward pass.

        The Qwen chat template ends its assistant-generation prefix with a
        newline, so each option is tokenized as a standalone completion.  A
        strict one-token requirement makes all option losses directly
        comparable and avoids length normalization or free-form parsing.
        """

        if not isinstance(options, Mapping) or not options:
            raise ValueError('options must be a non-empty mapping')
        self._load()
        token_ids = {}
        seen_token_ids = set()
        for label, completion in options.items():
            normalized_label = str(label)
            if not normalized_label:
                raise ValueError('option labels must not be empty')
            if not isinstance(completion, str) or not completion:
                raise ValueError('option completions must be non-empty strings')
            ids = self._processor.tokenizer.encode(
                completion,
                add_special_tokens=False,
            )
            if len(ids) != 1:
                raise ValueError(
                    f'option {normalized_label!r} must tokenize to exactly '
                    f'one token, got {ids}'
                )
            token_id = int(ids[0])
            if token_id in seen_token_ids:
                raise ValueError(
                    f'option {normalized_label!r} reuses token id {token_id}'
                )
            seen_token_ids.add(token_id)
            token_ids[normalized_label] = token_id

        inputs = self._processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        )
        inputs = inputs.to(self._input_device)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError('PyTorch is required for option scoring') from error
        with torch.inference_mode():
            outputs = self._model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
            next_token_log_probabilities = torch.log_softmax(
                outputs.logits[0, -1, :].float(),
                dim=-1,
            )
        losses = {
            label: float(-next_token_log_probabilities[token_id].item())
            for label, token_id in token_ids.items()
        }
        return SingleTokenOptionScores(
            negative_log_likelihoods=losses,
            token_ids=token_ids,
        )
