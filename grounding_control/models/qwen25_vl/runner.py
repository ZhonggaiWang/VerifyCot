"""Reusable lazy local Qwen2.5-VL model runner.

The imports are intentionally delayed so the original VoCoT environment can
import verifier modules without requiring a newer Transformers installation.
"""

from typing import Any, Mapping, Optional, Protocol, Sequence

DEFAULT_MIN_PIXELS = 4 * 28 * 28
# ``None`` means no project-level image-pixel cap.  The runner raises the
# processor's maximum dynamically to accommodate the actual PIL images in
# each request.  This deliberately preserves full source-image resolution.
DEFAULT_MAX_PIXELS = None
_QWEN_IMAGE_FACTOR = 28


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
            max_pixels: Optional[int] = DEFAULT_MAX_PIXELS,
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
        if max_pixels is not None and (
                not isinstance(max_pixels, int)
                or isinstance(max_pixels, bool)
                or max_pixels <= 0):
            raise ValueError('max_pixels must be a positive integer or None')
        if max_pixels is not None and min_pixels > max_pixels:
            raise ValueError('min_pixels must not exceed max_pixels')
        self.model_path = str(model_path)
        self.device = str(device)
        self.dtype = str(dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.min_pixels = int(min_pixels)
        self.max_pixels = (
            None if max_pixels is None else int(max_pixels)
        )
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
        processor_kwargs = {
            'local_files_only': self.local_files_only,
            'use_fast': False,
            'min_pixels': self.min_pixels,
        }
        if self.max_pixels is not None:
            processor_kwargs['max_pixels'] = self.max_pixels
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            **processor_kwargs,
        )
        self._input_device = next(self._model.parameters()).device

    @staticmethod
    def _message_image_max_pixels(
            messages: Sequence[Mapping[str, Any]]) -> Optional[int]:
        """Return the factor-rounded area needed to retain all PIL images."""
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - PIL is a runtime dependency.
            return None
        maximum = 0
        for message in messages:
            content = message.get('content', ())
            if not isinstance(content, (list, tuple)):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                image = part.get('image')
                if not isinstance(image, Image.Image):
                    continue
                width, height = image.size
                resized_width = max(
                    _QWEN_IMAGE_FACTOR,
                    round(width / _QWEN_IMAGE_FACTOR) * _QWEN_IMAGE_FACTOR,
                )
                resized_height = max(
                    _QWEN_IMAGE_FACTOR,
                    round(height / _QWEN_IMAGE_FACTOR) * _QWEN_IMAGE_FACTOR,
                )
                maximum = max(maximum, resized_width * resized_height)
        return maximum or None

    def generate(self, messages: Sequence[Mapping[str, Any]]) -> str:
        self._load()
        if self.max_pixels is None:
            # ``AutoProcessor`` otherwise restores the checkpoint config's
            # finite maximum.  Raising it per request implements the explicit
            # no-project-cap policy while retaining Qwen's factor alignment.
            effective_max_pixels = self._message_image_max_pixels(messages)
            if effective_max_pixels is not None:
                self._processor.image_processor.max_pixels = max(
                    self.min_pixels, effective_max_pixels
                )
        inputs = self._processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        )
        inputs = inputs.to(self._input_device)
        # Transformers 4.49's Qwen2.5-VL forward computes vocabulary logits
        # for every prompt position, although autoregressive generation only
        # consumes the final position.  An uncapped high-resolution image can
        # therefore create a multi-GiB [batch, sequence, vocabulary] tensor
        # after all visual reasoning has already completed.  Restricting the
        # LM head input to the final hidden state is generation-equivalent and
        # does not remove, resize, or otherwise limit any visual token.
        lm_head = getattr(self._model, 'lm_head', None)
        last_logit_hook = None
        if lm_head is not None:
            def _keep_last_hidden_state(_module, args):
                if not args or getattr(args[0], 'ndim', 0) < 3:
                    return args
                return (args[0][:, -1:, :],) + tuple(args[1:])

            last_logit_hook = lm_head.register_forward_pre_hook(
                _keep_last_hidden_state
            )
        try:
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        finally:
            if last_logit_hook is not None:
                last_logit_hook.remove()
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
