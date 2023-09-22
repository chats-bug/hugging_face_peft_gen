import torch
from transformers import (
    AutoTokenizer,
	AutoModelForCausalLM
)
from typing import List, Union


class HfModel:
    def __init__(
        self,
        model_id: str,
        quantization_config=None,
        device_map="auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        self.model_id = model_id
        self.quantization_config = quantization_config
        self.device_map = device_map

        self.model_loaded = False
        self._load_model_and_tokenizer(load_in_4bit, load_in_8bit, **kwargs)

    def _load_model_and_tokenizer(
        self, load_in_4bit: bool = False, load_in_8bit: bool = False, **kwargs
    ):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                device_map=self.device_map,
                quantization_config=self.quantization_config,
                **kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            # Suppress fast_tokenizer warning
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

            self.model_loaded = True
        except Exception as e:
            print(f"Error loading the model: \n{e}")
            self.model = None
            self.tokenizer = None
            self.model_loaded = False

    def generate(
        self,
        message: Union[str, List[str]],
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 0.7,
        num_return_sequences: int = 1,
        device="auto",
    ) -> List[str]:
        assert self.model_loaded, "Model and tokenizer not loaded properly"

        # Generation Config: this is the new standard way to do it in HF
        generation_config = self.model.generation_config
        generation_config.max_new_tokens = max_new_tokens
        generation_config.temperature = temperature
        generation_config.top_p = top_p
        generation_config.num_return_sequences = num_return_sequences
        generation_config.pad_token_id = self.tokenizer.eos_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id

        encoding = self.tokenizer(message, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                generation_config=generation_config,
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
