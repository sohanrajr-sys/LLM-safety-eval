import gc
import json
from pathlib import Path
from typing import List

CHAT_TEMPLATES = {
    "tinyllama": lambda text: (
        f"<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{text}</s>\n<|assistant|>\n"
    ),
    "phi2": lambda text: f"Instruct: {text}\nOutput:",
    "mistral": lambda text: f"<s>[INST] {text} [/INST]",
    "zephyr": lambda text: (
        f"<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{text}</s>\n<|assistant|>\n"
    ),
    "vicuna": lambda text: (
        "A chat between a curious user and an AI assistant. "
        "The assistant gives helpful, detailed, and polite answers.\n\n"
        f"USER: {text}\nASSISTANT:"
    ),
}


def format_prompt(text: str, chat_template: str) -> str:
    if chat_template not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown chat template: {chat_template!r}")
    return CHAT_TEMPLATES[chat_template](text)


class CheckpointManager:
    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path
        self._completed: set = set()
        if checkpoint_path.exists():
            data = json.loads(checkpoint_path.read_text())
            self._completed = set(data.get("completed", []))

    def is_done(self, model_slug: str, prompt_id: str) -> bool:
        return f"{model_slug}:{prompt_id}" in self._completed

    def mark_done(self, model_slug: str, prompt_ids: List[str]) -> None:
        for pid in prompt_ids:
            self._completed.add(f"{model_slug}:{pid}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"completed": list(self._completed)}, indent=2)
        )


class ModelRunner:
    def __init__(
        self,
        model_id: str,
        chat_template: str,
        batch_size: int,
        quantize: bool = False,
    ):
        self.model_id = model_id
        self.chat_template = chat_template
        self.batch_size = batch_size
        self.quantize = quantize
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="cuda:0",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="cuda:0",
            )
        self.model.eval()

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        seed: int = 42,
        max_new_tokens: int = 256,
    ) -> List[str]:
        import torch

        torch.manual_seed(seed)
        formatted = [format_prompt(p, self.chat_template) for p in prompts]
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda:0")

        with torch.no_grad():
            if temperature <= 0.01:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        input_len = inputs["input_ids"].shape[1]
        responses = []
        for output in outputs:
            new_tokens = output[input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(text.strip())
        return responses

    def unload(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
