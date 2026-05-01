"""
rlhf_loop.py — RLAIF: Reinforcement Learning from AI (Constitutional) Feedback.

Maps to Section 4–5 of Bai et al. (2022):

  * We use PPO (Proximal Policy Optimization) to fine-tune Mistral-7B
    against the reward model trained in reward_model.py.

  * The policy is initialized from the SFT checkpoint (or the base model
    if no SFT data is available).

  * A frozen reference model provides the KL penalty:
        reward_adjusted = r_rm(x, y) - β * KL(π_θ || π_ref)
    This prevents the policy from collapsing to reward-hacking gibberish.

  * Implementation uses trl.PPOTrainer, which handles the rollout buffer,
    advantage estimation (GAE), and the clipped PPO objective.

Key design decisions
---------------------
* We use PEFT LoRA on the policy backbone so only ~0.3% of parameters are
  trained, making PPO feasible on a single GPU.
* The reward model is loaded in inference-only mode (no gradients).
* We log PPO diagnostics (policy loss, value loss, KL, mean reward) to
  MLflow at each step.
* Batch size is kept small (4–8) to avoid OOM on 24 GB; gradient
  accumulation compensates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import mlflow
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RLAIFConfig:
    """Hyperparameters for the PPO / RLAIF training loop."""

    # Models
    policy_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    reward_model_dir: str = "checkpoints/reward_model"
    sft_checkpoint: str = ""            # optional SFT init; empty = use base

    # Quantization
    load_in_4bit: bool = True

    # LoRA for policy
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Mistral attention projection names
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Data
    prompt_file: str = "data/red_teaming_prompts.jsonl"
    max_prompt_length: int = 256
    max_new_tokens: int = 256

    # PPO
    output_dir: str = "checkpoints/ppo_policy"
    ppo_epochs: int = 4               # number of PPO gradient steps per batch
    num_rollout_steps: int = 512      # total environment steps per outer iter
    batch_size: int = 8               # rollout batch size
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    kl_penalty: str = "kl"            # "kl" | "abs" | "mse" | "full"
    init_kl_coef: float = 0.2         # β in KL penalty
    target_kl: float = 6.0            # adaptive KL target
    gamma: float = 1.0
    lam: float = 0.95                 # GAE lambda
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    seed: int = 42
    num_outer_iterations: int = 50    # outer PPO iterations

    # Generation (rollout)
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1

    # Logging
    mlflow_experiment: str = "constitutional-ai-ppo"
    mlflow_run_name: str = "rlaif-v1"
    log_every_n_steps: int = 10


# ──────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_policy(config: RLAIFConfig) -> AutoModelForCausalLMWithValueHead:
    """
    Load the policy model with a value head for PPO.

    trl.AutoModelForCausalLMWithValueHead wraps any causal LM and adds a
    scalar value head on top.  We apply LoRA to the underlying LM before
    wrapping so only adapter weights are updated by PPO.
    """
    base_name = config.sft_checkpoint or config.policy_model
    bnb = _bnb_config() if config.load_in_4bit else None

    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if config.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        base = get_peft_model(base, lora_cfg)
        base.print_trainable_parameters()

    return AutoModelForCausalLMWithValueHead.from_pretrained(base)


def load_ref_model(config: RLAIFConfig) -> AutoModelForCausalLMWithValueHead:
    """
    Load a frozen reference model for the KL penalty.

    The reference model must be identical to the initial policy.  We load
    it separately (no LoRA) and freeze all parameters.
    """
    base_name = config.sft_checkpoint or config.policy_model
    bnb = _bnb_config() if config.load_in_4bit else None

    ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def load_reward_model(config: RLAIFConfig):
    """Load the pre-trained reward model in inference-only mode."""
    rm_dir = Path(config.reward_model_dir)
    if not rm_dir.exists():
        raise FileNotFoundError(
            f"Reward model not found at {rm_dir}. "
            "Run `python reward_model.py` first."
        )
    rm = AutoModelForSequenceClassification.from_pretrained(
        str(rm_dir),
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    return rm


# ──────────────────────────────────────────────────────────────────────────────
# Reward function
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def score_responses(
    prompts: List[str],
    responses: List[str],
    rm,
    rm_tokenizer,
    max_length: int = 512,
) -> List[float]:
    """
    Score (prompt, response) pairs with the reward model.

    Returns a list of scalar reward values (one per example).
    """
    texts = [f"Human: {p}\n\nAssistant: {r}" for p, r in zip(prompts, responses)]
    enc = rm_tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(rm.device)

    logits = rm(**enc).logits  # (B, 1)
    return logits.squeeze(-1).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def load_prompts_dataset(
    prompt_file: str, tokenizer, max_prompt_length: int
) -> Dataset:
    import json

    prompts = []
    with open(prompt_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])

    def tokenize(example):
        enc = tokenizer(
            f"[INST] {example['prompt']} [/INST]",
            max_length=max_prompt_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return {"input_ids": enc["input_ids"], "query": example["prompt"]}

    ds = Dataset.from_dict({"prompt": prompts})
    return ds.map(tokenize)


# ──────────────────────────────────────────────────────────────────────────────
# PPO Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(config: RLAIFConfig) -> None:
    torch.manual_seed(config.seed)

    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mlflow.log_params(
            {
                "policy_model": config.policy_model,
                "reward_model_dir": config.reward_model_dir,
                "kl_penalty": config.kl_penalty,
                "init_kl_coef": config.init_kl_coef,
                "ppo_epochs": config.ppo_epochs,
                "batch_size": config.batch_size,
                "lr": config.learning_rate,
            }
        )

        # ── Load models ───────────────────────────────────────────────────
        logger.info("Loading policy model …")
        policy = load_policy(config)

        logger.info("Loading reference model …")
        ref_model = load_ref_model(config)

        logger.info("Loading reward model …")
        rm = load_reward_model(config)

        # ── Tokenizers ────────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(config.policy_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        rm_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_dir)
        if rm_tokenizer.pad_token is None:
            rm_tokenizer.pad_token = rm_tokenizer.eos_token

        # ── Dataset ───────────────────────────────────────────────────────
        dataset = load_prompts_dataset(
            config.prompt_file, tokenizer, config.max_prompt_length
        )
        logger.info("Prompt dataset: %d examples", len(dataset))

        # ── PPO config ────────────────────────────────────────────────────
        ppo_config = PPOConfig(
            model_name=config.policy_model,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            mini_batch_size=config.mini_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            ppo_epochs=config.ppo_epochs,
            kl_penalty=config.kl_penalty,
            init_kl_coef=config.init_kl_coef,
            target=config.target_kl,
            gamma=config.gamma,
            lam=config.lam,
            cliprange=config.cliprange,
            cliprange_value=config.cliprange_value,
            vf_coef=config.vf_coef,
            seed=config.seed,
        )

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=policy,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=lambda data: dict(
                (key, [d[key] for d in data]) for key in data[0]
            ),
        )

        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # ── Outer training loop ───────────────────────────────────────────
        logger.info("Starting PPO training …")
        step = 0

        for outer_iter in range(config.num_outer_iterations):
            for batch in ppo_trainer.dataloader:
                query_tensors = batch["input_ids"]

                # Rollout: generate responses from the current policy
                response_tensors = ppo_trainer.generate(
                    query_tensors, return_prompt=False, **gen_kwargs
                )

                # Decode for reward scoring
                queries_text = tokenizer.batch_decode(
                    query_tensors, skip_special_tokens=True
                )
                responses_text = tokenizer.batch_decode(
                    response_tensors, skip_special_tokens=True
                )

                # Score with the reward model
                rewards_float = score_responses(
                    queries_text, responses_text, rm, rm_tokenizer
                )
                rewards = [torch.tensor(r) for r in rewards_float]

                # PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

                if step % config.log_every_n_steps == 0:
                    mean_reward = sum(rewards_float) / len(rewards_float)
                    logger.info(
                        "iter=%d step=%d mean_reward=%.4f kl=%.4f policy_loss=%.4f",
                        outer_iter,
                        step,
                        mean_reward,
                        stats.get("objective/kl", float("nan")),
                        stats.get("ppo/loss/policy", float("nan")),
                    )
                    mlflow.log_metrics(
                        {
                            "mean_reward": mean_reward,
                            "kl": stats.get("objective/kl", 0.0),
                            "policy_loss": stats.get("ppo/loss/policy", 0.0),
                            "value_loss": stats.get("ppo/loss/value", 0.0),
                        },
                        step=step,
                    )

                step += 1

        # ── Save ──────────────────────────────────────────────────────────
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ppo_trainer.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        logger.info("PPO policy saved to %s", out_dir)
        mlflow.log_artifact(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RLAIF PPO loop.")
    parser.add_argument("--policy-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--reward-model-dir", default="checkpoints/reward_model")
    parser.add_argument("--prompt-file", default="data/red_teaming_prompts.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/ppo_policy")
    parser.add_argument("--outer-iters", type=int, default=50)
    parser.add_argument("--kl-coef", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    cfg = RLAIFConfig(
        policy_model=args.policy_model,
        reward_model_dir=args.reward_model_dir,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        num_outer_iterations=args.outer_iters,
        init_kl_coef=args.kl_coef,
        learning_rate=args.lr,
        load_in_4bit=not args.no_4bit,
    )
    train(cfg)
