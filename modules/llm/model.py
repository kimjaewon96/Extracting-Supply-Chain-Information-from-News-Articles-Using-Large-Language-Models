import gc
from typing import TYPE_CHECKING, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    GenerationConfig,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class LLMModel:
    def __init__(self, model=None, tokenizer=None, generation_config: dict = {}):
        self.model: "PreTrainedModel" = model
        self.tokenizer: "PreTrainedTokenizer" = tokenizer
        self.cache = None
        self.loader: str = None
        self.default_generation_config = generation_config

    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.generation_config = None
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def from_transformers(
        cls,
        model_name: str,
        max_length: Optional[int] = None,
        device: str = "auto",
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
    ):
        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = device
        model_kwargs.setdefault("torch_dtype", torch.bfloat16)
        # model_kwargs.setdefault("attn_implementation", "flash_attention")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer_kwargs.setdefault("padding_side", "left")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        generation_config = {
            "bos_token_id": model.generation_config.bos_token_id,
            "eos_token_id": model.generation_config.eos_token_id,
            "do_sample": True,
            "max_length": max_length or model.generation_config.max_length,
        }
        if model.generation_config.pad_token_id:
            generation_config["pad_token_id"] = model.generation_config.pad_token_id
        else:
            if isinstance(model.generation_config.eos_token_id, list):
                generation_config["pad_token_id"] = (
                    model.generation_config.eos_token_id[0]
                )
            else:
                generation_config["pad_token_id"] = model.generation_config.eos_token_id
        return cls(model, tokenizer, generation_config)

    def get_generation_config(self, custom_generation_config: Optional[dict] = {}):
        generation_config = {
            **self.default_generation_config,
            **custom_generation_config,
            "return_dict_in_generate": True,
            "use_cache": True,
        }
        return GenerationConfig(**generation_config)

    @torch.inference_mode()
    def generate(self, prompt: str, custom_generation_config: Optional[dict] = {}):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.model.device)
        input_seq_len = len(inputs.input_ids[0])
        generation_config = self.get_generation_config(custom_generation_config)
        # generation_config.do_sample = False
        output = self.model.generate(
            **inputs,
            tokenizer=self.tokenizer,
            logits_processor=None,
            generation_config=generation_config,
            past_key_values=DynamicCache(),
        )
        output_ids = output.sequences
        output_text = self.tokenizer.batch_decode(output_ids[:, input_seq_len:-1])[0]
        return output_text

    @torch.inference_mode()
    def choice(self, prompt: str, choices: list[str] = []):
        assert len(choices) >= 2
        input_prompt = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.model.device)
        assert all(
            [
                len(x) == 1
                for x in self.tokenizer(choices, add_special_tokens=False).input_ids
            ]
        )
        labels = self.tokenizer(
            choices, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.model.device)
        outputs = self.model(**input_prompt, past_key_values=DynamicCache())
        logits = outputs.logits[..., -1, :]
        loss_fct = torch.nn.CrossEntropyLoss()
        logits = logits.view(-1, self.model.config.vocab_size)
        scores = []
        for label in labels:
            loss = loss_fct(logits, label).item()
            scores.append(loss)
        min_idx = scores.index(min(scores))
        return choices[min_idx]
