"""Training / fine-tuning planner and runner.

Two sides of the same coin:

* :class:`TrainingPlanner` — pre-flight memory & compute estimator. Pure
  Python, no torch. Tells you whether a run will fit, roughly how long it
  will take, and roughly what it will cost, before you install CUDA.
* :mod:`unsloth_runner` — real training backend. Requires the ``[train]``
  extras (Unsloth, TRL, PEFT, datasets, transformers, torch). Delivers
  2–5× speed and 70 % less VRAM than vanilla transformers training.

Usage::

    from kv_planner.core.training import TrainingPlanner
    plan = TrainingPlanner().plan(model, gpu, method="qlora", ...)
    # inspect plan.total_memory_gb, plan.est_training_hours, plan.est_cost_usd

    from kv_planner.core.training.unsloth_runner import TrainArgs, run_training
    run_training(TrainArgs(model_slug_or_id="llama-3-8b",
                           dataset="./my_data.jsonl",
                           output_dir="./ft-out",
                           method="qlora"))
"""

from kv_planner.core.training.planner import (
    TrainingMethod,
    TrainingPlan,
    TrainingPlanner,
)

__all__ = [
    "TrainingMethod",
    "TrainingPlan",
    "TrainingPlanner",
]
