"""
reward_model.py — Train a reward model (RM) on constitutional preference pairs.

Maps to Section 4 of Bai et al. (2022) — "RL-CAI":

  * Input: JSONL triplets from critique_revise.py
    Each triplet provides a preference pair:
        chosen  = revised_response   (constitutionally improved)
        rejected = original_response (potentially harmful)

  * Architecture: We take a pretrained causal LM (GPT-2 or a 4-bit
    Mistral) and add a scalar value head on the final hidden state,
    following the TRL RewardTrainer convention.

  * Loss: Bradley-Terry pairwise ranking loss (standard in RLHF):
        L = -E[log σ(r_chosen - r_rejected)]
    where r = reward head output on the last non-padding token.

  * Logging: MLflow experiment tracking with loss curves and eval metrics.

Design decisions
----------------
* We use GPT-2 as the default RM backbone for fast iteration.  Swap
  `rm_base_model` to "mistralai/Mistral-7B-Instruct-v0.2" + load_in_4bit
  for higher quality at the cost of ~8 GB VRAM.
* We use PEFT (LoRA) on top of the frozen backbone so we only train a few
  million parameters — critical when the backbone is a 7B model.
* The RewardTrainer from trl handles the Bradley-Terry loss and evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import RewardTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RMConfig:
    """Hyperparameters for reward model training."""

    # Model
    rm_base_model: str = "gpt2"       # or "mistralai/Mistral-7B-Instruct-v0.2"
    load_in_4bit: bool = False         # set True when using Mistral backbone
    use_lora: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj"]  # GPT-2 attention
    )

    # Data
    triplets_file: str = "data/critique_revise_triplets.jsonl"
    val_split: float = 0.1
    max_length: int = 512

    # Training
    output_dir: str = "checkpoints/reward_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    seed: int = 42

    # MLflow
    mlflow_experiment: str = "constitutional-ai-reward-model"
    mlflow_run_name: str = "rm-v1"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_preference_dataset(
    triplets_file: str,
    tokenizer,
    max_length: int,
    val_split: float,
) -> Tuple[Dataset, Dataset]:
    """
    Convert JSONL triplets → HuggingFace Dataset with columns:
        input_ids_chosen, attention_mask_chosen,
        input_ids_rejected, attention_mask_rejected

    RewardTrainer expects exactly this schema.
    """
    path = Path(triplets_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Triplets file not found: {path}\n"
            "Run `python critique_revise.py` first."
        )

    chosen_texts: List[str] = []
    rejected_texts: List[str] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj["prompt"]
            # Revised (constitutionally improved) is preferred
            chosen_texts.append(_format_pair(prompt, obj["revised_response"]))
            rejected_texts.append(_format_pair(prompt, obj["original_response"]))

    logger.info("Loaded %d preference pairs", len(chosen_texts))

    # Tokenize
    def tokenize(texts: List[str], label: str) -> Dict:
        enc = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # return lists for Dataset
        )
        return {
            f"input_ids_{label}": enc["input_ids"],
            f"attention_mask_{label}": enc["attention_mask"],
        }

    chosen_enc = tokenize(chosen_texts, "chosen")
    rejected_enc = tokenize(rejected_texts, "rejected")

    raw = {**chosen_enc, **rejected_enc}
    dataset = Dataset.from_dict(raw)

    split = dataset.train_test_split(test_size=val_split, seed=42)
    return split["train"], split["test"]


def _format_pair(prompt: str, response: str) -> str:
    """Format a (prompt, response) pair as a single string for the RM."""
    return f"Human: {prompt}\n\nAssistant: {response}"


# ──────────────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────────────

def build_reward_model(config: RMConfig):
    """
    Build a sequence-classification model with num_labels=1 as the reward head.

    AutoModelForSequenceClassification adds a linear layer on top of the
    pooled hidden state and returns a scalar reward per sequence.
    We follow the same pattern as the TRL RewardTrainer examples.
    """
    bnb_config = None
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.rm_base_model,
        num_labels=1,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )

    # Fix: GPT-2 has no pad token by default
    tokenizer = AutoTokenizer.from_pretrained(config.rm_base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if config.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(config: RMConfig) -> None:
    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mlflow.log_params(
            {
                "rm_base_model": config.rm_base_model,
                "use_lora": config.use_lora,
                "lora_r": config.lora_r,
                "num_epochs": config.num_train_epochs,
                "lr": config.learning_rate,
                "batch_size": config.per_device_train_batch_size,
                "max_length": config.max_length,
            }
        )

        model, tokenizer = build_reward_model(config)

        train_ds, eval_ds = load_preference_dataset(
            config.triplets_file, tokenizer, config.max_length, config.val_split
        )
        logger.info(
            "Train size: %d | Eval size: %d", len(train_ds), len(eval_ds)
        )

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            fp16=config.fp16,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            eval_strategy=config.eval_strategy,
            save_strategy=config.save_strategy,
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
            seed=config.seed,
            report_to=[],  # we handle logging via MLflow manually
        )

        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
        )

        logger.info("Starting reward model training …")
        trainer.train()

        # Log final metrics
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        logger.info("Eval metrics: %s", metrics)

        # Save
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        logger.info("Reward model saved to %s", config.output_dir)

        mlflow.log_artifact(config.output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Constitutional AI reward model.")
    parser.add_argument("--rm-base-model", default="gpt2")
    parser.add_argument("--triplets-file", default="data/critique_revise_triplets.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/reward_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    cfg = RMConfig(
        rm_base_model=args.rm_base_model,
        triplets_file=args.triplets_file,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        use_lora=not args.no_lora,
        load_in_4bit=args.load_in_4bit,
    )
    train(cfg)
