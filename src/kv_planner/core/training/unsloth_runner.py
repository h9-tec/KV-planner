"""Real training backend: Unsloth + TRL.

Unsloth (https://github.com/unslothai/unsloth) patches Transformers to
deliver ~2× training speed and ~70 % VRAM reduction for LoRA / QLoRA
fine-tunes on Llama / Qwen / Mistral / Gemma / Phi families. It wraps
``torch.compile``, a custom CUDA RoPE kernel, and memory-efficient
attention; we drive it through the standard TRL ``SFTTrainer``/``DPOTrainer``
APIs so users aren't locked into us.

This module **imports Unsloth lazily** so ``kv_planner`` keeps loading
even if the user hasn't installed the ``[train]`` extras. Call
:func:`ensure_available` before anything else to get a friendly error.

Install::

    pip install -e '.[train]'

Supported pipelines:

* **SFT** — instruction tuning on ShareGPT / Alpaca / ChatML JSONL
* **DPO** — preference optimization on (prompt, chosen, rejected) JSONL
* **GRPO** — group-relative policy optimization (R1-style reasoning)
* **LoRA / QLoRA** via adapter rank, target-module selection, and 4-bit base
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)

Pipeline = Literal["sft", "dpo", "grpo"]
Method = Literal["lora", "qlora", "full_ft"]


# Unsloth ships pre-quantized backbones under its own HF org.
# Map our catalog slugs to Unsloth-preferred HF ids where the 4-bit version
# is available. Where not mapped, fall back to the canonical HF id.
UNSLOTH_MODEL_MAP: dict[str, str] = {
    "llama-3.2-1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "llama-3-8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama-3-70b": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
    "qwen2.5-3b": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "qwen2.5-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "qwen2.5-14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "qwen2.5-coder-7b": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    "qwen2.5-coder-14b": "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
    "mistral-7b-v0.3": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "phi-4-14b": "unsloth/phi-4-bnb-4bit",
    "deepseek-r1-distill-7b": "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
    "deepseek-r1-distill-14b": "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
    "deepseek-r1-distill-32b": "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
}


@dataclass
class TrainArgs:
    """User-facing training config. Defaults tuned for single-GPU QLoRA."""

    model_slug_or_id: str
    dataset: str
    output_dir: str
    pipeline: Pipeline = "sft"
    method: Method = "qlora"

    # Core training hyperparams
    max_seq_length: int = 2048
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    seed: int = 3407
    logging_steps: int = 10
    save_steps: int = 200
    optim: str = "adamw_8bit"

    # LoRA hyperparams
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    use_gradient_checkpointing: str = "unsloth"

    # Dataset field/format for SFT
    dataset_text_field: str = "text"
    chat_template: Optional[str] = None  # e.g., "llama-3.1" / "qwen-2.5" / "chatml"

    # Misc
    report_to: str | None = "none"  # "tensorboard" / "wandb" / "none"
    push_to_hub_id: str | None = None
    hf_token: str | None = None

    # For DPO / GRPO
    beta: float = 0.1


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------


REQUIRED_PACKAGES = ("unsloth", "trl", "peft", "datasets", "transformers", "torch")


def ensure_available() -> None:
    """Raise :class:`ImportError` with a helpful message if the training
    extras aren't installed."""
    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        raise ImportError(
            "kv-planner training backend requires "
            + ", ".join(missing)
            + ".  Install via:  pip install -e '.[train]'"
            " (requires CUDA + PyTorch 2.2+)."
        )


def is_available() -> bool:
    try:
        ensure_available()
    except ImportError:
        return False
    return True


def resolve_model_id(slug_or_id: str, prefer_4bit: bool = True) -> str:
    """Map a catalog slug to an Unsloth-preferred model ID when possible."""
    if prefer_4bit and slug_or_id in UNSLOTH_MODEL_MAP:
        return UNSLOTH_MODEL_MAP[slug_or_id]
    return slug_or_id


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    output_dir: str
    final_train_loss: float | None
    steps_run: int
    seconds_elapsed: float
    pushed_to_hub: str | None


def run_training(
    args: TrainArgs,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Launch a full Unsloth + TRL training job.

    Progress is reported via ``progress_cb`` once per ``logging_steps``; the
    callback receives a dict of ``{'step', 'loss', 'lr', 'epoch'}``.
    """
    ensure_available()

    import time
    import torch
    from datasets import Dataset, load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    model_id = resolve_model_id(args.model_slug_or_id)
    load_in_4bit = args.method == "qlora"
    dtype = None if load_in_4bit else (torch.bfloat16 if is_bfloat16_supported() else torch.float16)

    logger.info("Loading %s (4bit=%s) …", model_id, load_in_4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=args.hf_token or os.environ.get("HF_TOKEN"),
    )

    if args.method in ("lora", "qlora"):
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=args.lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )

    # Apply a chat template if requested (Unsloth ships helpers).
    if args.chat_template:
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    # --- Dataset ---------------------------------------------------------
    dataset = _load_dataset(args.dataset, args.dataset_text_field, tokenizer, args.chat_template)

    # --- Progress callback ----------------------------------------------
    cb_list: list[TrainerCallback] = []
    if progress_cb is not None:
        class _CB(TrainerCallback):
            def on_log(self, _a, _b, _c, logs=None, **_kw):  # type: ignore[override]
                if logs:
                    progress_cb(dict(logs))
        cb_list.append(_CB())

    # --- Pipeline dispatch ----------------------------------------------
    t_start = time.perf_counter()
    if args.pipeline == "sft":
        sft_cfg = SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            report_to=args.report_to or "none",
            dataset_text_field=args.dataset_text_field,
            max_length=args.max_seq_length,
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=sft_cfg,
            callbacks=cb_list,
        )
    elif args.pipeline == "dpo":
        from trl import DPOConfig, DPOTrainer
        dpo_cfg = DPOConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            report_to=args.report_to or "none",
            beta=args.beta,
            max_length=args.max_seq_length,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=dpo_cfg,
            callbacks=cb_list,
        )
    elif args.pipeline == "grpo":
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
        grpo_cfg = GRPOConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            report_to=args.report_to or "none",
            beta=args.beta,
            max_prompt_length=args.max_seq_length,
        )
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=grpo_cfg,
            callbacks=cb_list,
        )
    else:
        raise ValueError(f"Unknown pipeline {args.pipeline}")

    trainer_stats = trainer.train()
    elapsed = time.perf_counter() - t_start

    # Persist the PEFT adapter (LoRA/QLoRA) and tokenizer.
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir / "adapter"))
    tokenizer.save_pretrained(str(save_dir / "adapter"))

    pushed = None
    if args.push_to_hub_id:
        model.push_to_hub(args.push_to_hub_id, token=args.hf_token)
        tokenizer.push_to_hub(args.push_to_hub_id, token=args.hf_token)
        pushed = args.push_to_hub_id

    return TrainResult(
        output_dir=str(save_dir),
        final_train_loss=float(trainer_stats.training_loss) if hasattr(trainer_stats, "training_loss") else None,
        steps_run=int(trainer_stats.global_step) if hasattr(trainer_stats, "global_step") else 0,
        seconds_elapsed=elapsed,
        pushed_to_hub=pushed,
    )


# ---------------------------------------------------------------------------
# Dataset loading — supports JSONL, HF Hub, and chat-formatted files.
# ---------------------------------------------------------------------------


def _load_dataset(path: str, text_field: str, tokenizer, chat_template: str | None):
    """Load a dataset from an HF Hub id or a local file.

    Accepted formats:
    * ``file.jsonl`` — one sample per line (SFT). If records have a
      ``messages`` field, the chat template is applied and the rendered
      string is placed in ``text_field``.
    * ``file.json`` — list of records
    * any HF dataset id — passed through to :func:`datasets.load_dataset`
    """
    from datasets import Dataset, load_dataset

    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in (".jsonl", ".ndjson"):
            ds = load_dataset("json", data_files=str(p), split="train")
        elif p.suffix.lower() == ".json":
            ds = load_dataset("json", data_files=str(p), split="train")
        else:
            raise ValueError(f"Unsupported dataset file extension: {p.suffix}")
    else:
        ds = load_dataset(path, split="train")

    # If ``messages`` column is present, render with the chat template.
    if "messages" in ds.column_names:
        def _fmt(ex):
            ex[text_field] = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            return ex
        ds = ds.map(_fmt)

    return ds
