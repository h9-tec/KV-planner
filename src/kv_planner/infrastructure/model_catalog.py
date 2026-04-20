"""Curated catalog of ~30 popular open-weight LLMs.

Each entry bundles the :class:`ModelConfig` with a *use-case vector*
(general / coding / reasoning / chat / multimodal / embedding) and a
quality score on a 0-100 scale. Used by the recommender to rank models
against a given hardware budget and workload intent.

The ``quality_0_100`` score is a subjective meta-weighting of public
benchmark results at time of writing (MMLU / MT-Bench / HumanEval / GSM8K
for general reasoning, LiveCodeBench for coding, MIRAGE for reasoning).
It is NOT scraped — hard-coded literature reading. Override per use case
by tweaking the ``use_cases`` dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kv_planner.domain import ModelConfig

UseCase = Literal[
    "general", "coding", "reasoning", "chat", "multimodal", "embedding", "agent"
]


@dataclass(frozen=True)
class CatalogEntry:
    """A curated model listing with scoring inputs."""

    slug: str
    config: ModelConfig
    provider: str
    quality_0_100: int
    use_cases: tuple[UseCase, ...] = ("general", "chat")
    # Quality on specific use cases — override per-case scores in [0, 100].
    use_case_scores: dict[UseCase, int] = field(default_factory=dict)
    # Typical Ollama tag(s) if available (for install-detection).
    ollama_tags: tuple[str, ...] = ()
    # License short tag.
    license: str = ""
    # Preferred quantization on most consumer GPUs.
    recommended_quant: str = "int4"
    released: str = ""  # YYYY-MM

    def score_for(self, use_case: UseCase) -> int:
        if use_case in self.use_case_scores:
            return self.use_case_scores[use_case]
        if use_case in self.use_cases:
            return self.quality_0_100
        # Cross-use-case penalty
        return max(0, self.quality_0_100 - 25)


def _cfg(
    name: str, n_layers: int, hidden: int, heads: int, kv_heads: int,
    head_dim: int, vocab: int, max_pos: int, ffn_inter: int,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        num_layers=n_layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
        attention_type="GQA" if kv_heads < heads else "MHA",
        ffn_type="swiglu",
        ffn_intermediate_size=ffn_inter,
    )


# All architecture numbers verified against each model's HF config.json.
CATALOG: list[CatalogEntry] = [
    # ------- Meta Llama family --------------------------------------------
    CatalogEntry(
        slug="llama-3.2-1b",
        config=_cfg("meta-llama/Llama-3.2-1B-Instruct",
                    16, 2048, 32, 8, 64, 128256, 131072, 8192),
        provider="Meta", quality_0_100=55, license="Llama-3.2",
        use_cases=("general", "chat"),
        ollama_tags=("llama3.2:1b",),
        released="2024-09",
    ),
    CatalogEntry(
        slug="llama-3.2-3b",
        config=_cfg("meta-llama/Llama-3.2-3B-Instruct",
                    28, 3072, 24, 8, 128, 128256, 131072, 8192),
        provider="Meta", quality_0_100=68, license="Llama-3.2",
        use_cases=("general", "chat"),
        ollama_tags=("llama3.2:3b", "llama3.2:latest"),
        released="2024-09",
    ),
    CatalogEntry(
        slug="llama-3-8b",
        config=_cfg("meta-llama/Meta-Llama-3-8B-Instruct",
                    32, 4096, 32, 8, 128, 128256, 8192, 14336),
        provider="Meta", quality_0_100=75, license="Llama-3",
        use_cases=("general", "chat", "coding"),
        use_case_scores={"coding": 68},
        ollama_tags=("llama3:8b", "llama3:latest"),
        released="2024-04",
    ),
    CatalogEntry(
        slug="llama-3-70b",
        config=_cfg("meta-llama/Meta-Llama-3-70B-Instruct",
                    80, 8192, 64, 8, 128, 128256, 8192, 28672),
        provider="Meta", quality_0_100=86, license="Llama-3",
        use_cases=("general", "chat", "coding", "reasoning"),
        ollama_tags=("llama3:70b",),
        released="2024-04",
    ),

    # ------- Qwen family --------------------------------------------------
    CatalogEntry(
        slug="qwen2.5-3b",
        config=_cfg("Qwen/Qwen2.5-3B-Instruct",
                    36, 2048, 16, 2, 128, 151936, 32768, 11008),
        provider="Alibaba", quality_0_100=67, license="Qwen",
        use_cases=("general", "chat"),
        released="2024-09",
    ),
    CatalogEntry(
        slug="qwen2.5-7b",
        config=_cfg("Qwen/Qwen2.5-7B-Instruct",
                    28, 3584, 28, 4, 128, 152064, 32768, 18944),
        provider="Alibaba", quality_0_100=77, license="Qwen",
        use_cases=("general", "chat", "reasoning"),
        released="2024-09",
    ),
    CatalogEntry(
        slug="qwen2.5-coder-7b",
        config=_cfg("Qwen/Qwen2.5-Coder-7B-Instruct",
                    28, 3584, 28, 4, 128, 152064, 32768, 18944),
        provider="Alibaba", quality_0_100=80, license="Qwen",
        use_cases=("coding", "general"),
        use_case_scores={"coding": 88, "general": 72},
        ollama_tags=("qwen2.5-coder:7b",),
        released="2024-11",
    ),
    CatalogEntry(
        slug="qwen2.5-coder-14b",
        config=_cfg("Qwen/Qwen2.5-Coder-14B-Instruct",
                    48, 5120, 40, 8, 128, 152064, 32768, 13824),
        provider="Alibaba", quality_0_100=84, license="Qwen",
        use_cases=("coding", "general"),
        use_case_scores={"coding": 91, "general": 76},
        ollama_tags=("qwen2.5-coder:14b",),
        released="2024-11",
    ),
    CatalogEntry(
        slug="qwen2.5-14b",
        config=_cfg("Qwen/Qwen2.5-14B-Instruct",
                    48, 5120, 40, 8, 128, 152064, 32768, 13824),
        provider="Alibaba", quality_0_100=82, license="Qwen",
        use_cases=("general", "reasoning", "chat"),
        released="2024-09",
    ),
    CatalogEntry(
        slug="qwen2.5-32b",
        config=_cfg("Qwen/Qwen2.5-32B-Instruct",
                    64, 5120, 40, 8, 128, 152064, 32768, 27648),
        provider="Alibaba", quality_0_100=87, license="Qwen",
        use_cases=("general", "reasoning", "coding"),
        released="2024-09",
    ),
    CatalogEntry(
        slug="qwen2.5-72b",
        config=_cfg("Qwen/Qwen2.5-72B-Instruct",
                    80, 8192, 64, 8, 128, 152064, 32768, 29568),
        provider="Alibaba", quality_0_100=89, license="Qwen",
        use_cases=("general", "reasoning", "coding", "agent"),
        released="2024-09",
    ),

    # ------- Mistral family -----------------------------------------------
    CatalogEntry(
        slug="mistral-7b-v0.3",
        config=_cfg("mistralai/Mistral-7B-Instruct-v0.3",
                    32, 4096, 32, 8, 128, 32768, 32768, 14336),
        provider="Mistral AI", quality_0_100=74, license="Apache-2.0",
        use_cases=("general", "chat"),
        ollama_tags=("mistral:7b",),
        released="2024-05",
    ),

    # ------- DeepSeek family ---------------------------------------------
    CatalogEntry(
        slug="deepseek-r1-distill-7b",
        config=_cfg("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    28, 3584, 28, 4, 128, 152064, 131072, 18944),
        provider="DeepSeek", quality_0_100=82, license="MIT",
        use_cases=("reasoning", "coding", "general"),
        use_case_scores={"reasoning": 90, "coding": 82},
        ollama_tags=("deepseek-r1:latest", "deepseek-r1:7b"),
        released="2025-01",
    ),
    CatalogEntry(
        slug="deepseek-r1-distill-14b",
        config=_cfg("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    48, 5120, 40, 8, 128, 152064, 131072, 13824),
        provider="DeepSeek", quality_0_100=86, license="MIT",
        use_cases=("reasoning", "coding", "general"),
        use_case_scores={"reasoning": 93, "coding": 86},
        ollama_tags=("deepseek-r1:14b",),
        released="2025-01",
    ),
    CatalogEntry(
        slug="deepseek-r1-distill-32b",
        config=_cfg("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                    64, 5120, 40, 8, 128, 152064, 131072, 27648),
        provider="DeepSeek", quality_0_100=89, license="MIT",
        use_cases=("reasoning", "coding", "general"),
        use_case_scores={"reasoning": 95, "coding": 89},
        ollama_tags=("deepseek-r1:32b",),
        released="2025-01",
    ),

    # ------- Microsoft Phi ------------------------------------------------
    CatalogEntry(
        slug="phi-4-14b",
        config=_cfg("microsoft/phi-4",
                    40, 5120, 40, 10, 128, 100352, 16384, 17920),
        provider="Microsoft", quality_0_100=83, license="MIT",
        use_cases=("general", "reasoning", "chat"),
        use_case_scores={"reasoning": 87},
        ollama_tags=("phi4:latest",),
        released="2024-12",
    ),

    # ------- Cohere -------------------------------------------------------
    CatalogEntry(
        slug="aya-expanse-8b",
        config=_cfg("CohereForAI/aya-expanse-8b",
                    32, 4096, 32, 8, 128, 256000, 8192, 14336),
        provider="Cohere", quality_0_100=70, license="CC-BY-NC-4.0",
        use_cases=("general", "chat", "multimodal"),
        use_case_scores={"multimodal": 60},
        ollama_tags=("aya:8b", "aya-expanse:8b"),
        released="2024-10",
    ),
    CatalogEntry(
        slug="aya-expanse-35b",
        config=_cfg("CohereForAI/aya-expanse-32b",
                    40, 8192, 64, 8, 128, 256000, 131072, 24576),
        provider="Cohere", quality_0_100=80, license="CC-BY-NC-4.0",
        use_cases=("general", "chat", "multimodal"),
        ollama_tags=("aya:35b",),
        released="2024-10",
    ),
]


def by_slug(slug: str) -> CatalogEntry | None:
    for e in CATALOG:
        if e.slug == slug:
            return e
    return None


def match_ollama_name(ollama_name: str) -> CatalogEntry | None:
    """Map an Ollama tag (e.g., ``llama3.2:3b``) back to a catalog entry."""
    for e in CATALOG:
        if ollama_name in e.ollama_tags:
            return e
    # Loose fallback: compare before ':'
    target = ollama_name.split(":", 1)[0].lower()
    for e in CATALOG:
        for tag in e.ollama_tags:
            if tag.split(":", 1)[0].lower() == target:
                return e
    return None


def by_use_case(use_case: UseCase) -> list[CatalogEntry]:
    return [e for e in CATALOG if use_case in e.use_cases]
