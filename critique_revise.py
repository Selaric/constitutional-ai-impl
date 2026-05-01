"""
critique_revise.py — SL-CAI: Supervised Learning from Constitutional AI Feedback.

Maps to Section 3.1–3.3 of Bai et al. (2022):

  1. Generate an initial response to a (potentially harmful) prompt.
  2. Sample a principle from the constitution.
  3. Ask the model to critique its own response against that principle.
  4. Ask the model to revise the response given the critique.
  5. Repeat steps 2–4 for `num_rounds` critique–revision cycles.
  6. Save (prompt, original_response, critique, revised_response) triplets
     to a JSONL file for reward-model training.

The paper uses Claude as the generator; we substitute Mistral-7B-Instruct
with 4-bit NF4 quantization (bitsandbytes) so the pipeline runs on a
single 24 GB GPU.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from constitution import Constitution, Principle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CARConfig:
    """All hyperparameters for the critique-and-revise pipeline."""

    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    input_file: str = "data/red_teaming_prompts.jsonl"
    output_file: str = "data/critique_revise_triplets.jsonl"

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"         # nf4 | fp4
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # bfloat16 | float16

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1

    # Pipeline
    num_rounds: int = 1          # critique-revise cycles per example
    principles_per_example: int = 1  # how many principles to sample
    seed: int = 42
    device_map: str = "auto"


# ──────────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Triplet:
    """One (prompt, original, critique, revised) example."""

    prompt: str
    original_response: str
    principle_name: str
    critique: str
    revised_response: str
    # Round index (0-based); the paper chains multiple rounds
    round_idx: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class CritiqueReviser:
    """
    Loads Mistral-7B-Instruct and runs the CAI critique-revise pipeline.

    Memory footprint with 4-bit NF4:
        Mistral-7B params   ≈  7B × 0.5 bytes ≈  3.5 GB
        Activation buffers  ≈  2–4 GB
        Total               ≈  6–8 GB  (fits on a single 24 GB A100/3090)
    """

    def __init__(self, config: CARConfig) -> None:
        self.config = config
        self.constitution = Constitution()
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        logger.info("Loading tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model with 4-bit quantization …")
        self.model = self._load_quantized_model()
        self.model.eval()

        self.gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        logger.info("CritiqueReviser ready.")

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_quantized_model(self) -> AutoModelForCausalLM:
        compute_dtype = (
            torch.bfloat16
            if self.config.bnb_4bit_compute_dtype == "bfloat16"
            else torch.float16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            torch_dtype=compute_dtype,
            trust_remote_code=False,
        )

    # ── Text generation ────────────────────────────────────────────────────

    @torch.inference_mode()
    def _generate(self, prompt_text: str) -> str:
        """Tokenize `prompt_text`, run greedy/sampling decode, return completion."""
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
            **inputs,
            generation_config=self.gen_config,
        )

        # Strip the input prefix from the output
        new_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ── Pipeline steps ─────────────────────────────────────────────────────

    def _initial_response(self, prompt: str) -> str:
        """Step 1: elicit a (potentially harmful) initial response."""
        chat = f"[INST] {prompt} [/INST]"
        return self._generate(chat)

    def _critique(self, prompt: str, response: str, principle: Principle) -> str:
        """Step 2: ask the model to identify violations of the principle."""
        filled = principle.critique_prompt.format(
            prompt=prompt, response=response
        )
        return self._generate(filled)

    def _revise(
        self,
        prompt: str,
        response: str,
        critique: str,
        principle: Principle,
    ) -> str:
        """Step 3: ask the model to produce a revised, harmless response."""
        filled = principle.revision_prompt.format(
            prompt=prompt, response=response, critique=critique
        )
        return self._generate(filled)

    # ── Outer loop ─────────────────────────────────────────────────────────

    def process_example(self, prompt: str) -> List[Triplet]:
        """
        Run the full critique-revise pipeline for one prompt.

        Returns one Triplet per (round × principle) pair.
        The final revised response after all rounds is suitable for SFT.
        """
        triplets: List[Triplet] = []
        current_response = self._initial_response(prompt)
        logger.debug("Initial response generated (%d chars)", len(current_response))

        for round_idx in range(self.config.num_rounds):
            principles = self.constitution.sample(
                k=self.config.principles_per_example,
                seed=self.config.seed + round_idx,
            )
            for principle in principles:
                logger.debug(
                    "Round %d | Principle: %s", round_idx, principle.name
                )
                critique = self._critique(prompt, current_response, principle)
                revised = self._revise(prompt, current_response, critique, principle)
                triplets.append(
                    Triplet(
                        prompt=prompt,
                        original_response=current_response,
                        principle_name=principle.name,
                        critique=critique,
                        revised_response=revised,
                        round_idx=round_idx,
                    )
                )
                # Chain: next round starts from this revised response
                current_response = revised

        return triplets

    def run(self) -> None:
        """Process all prompts in `input_file`, write triplets to `output_file`."""
        input_path = Path(self.config.input_file)
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompts = _load_prompts(input_path)
        logger.info("Loaded %d prompts from %s", len(prompts), input_path)

        written = 0
        with output_path.open("w", encoding="utf-8") as fout:
            for i, prompt in enumerate(prompts):
                logger.info("[%d/%d] Processing prompt …", i + 1, len(prompts))
                t0 = time.time()
                try:
                    triplets = self.process_example(prompt)
                    for t in triplets:
                        fout.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
                        written += 1
                except Exception as exc:
                    logger.error("Failed on prompt %d: %s", i, exc)
                    continue
                logger.info(
                    "  → %d triplet(s) in %.1f s", len(triplets), time.time() - t0
                )

        logger.info("Done. Wrote %d triplets to %s", written, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_prompts(path: Path) -> List[str]:
    """Load prompts from a JSONL file (each line: {"prompt": "..."})."""
    prompts: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the CAI critique-revise pipeline.")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--input", default="data/red_teaming_prompts.jsonl")
    parser.add_argument("--output", default="data/critique_revise_triplets.jsonl")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = CARConfig(
        model_name=args.model,
        input_file=args.input,
        output_file=args.output,
        num_rounds=args.rounds,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    reviser = CritiqueReviser(cfg)
    reviser.run()
