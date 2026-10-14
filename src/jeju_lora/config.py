"""YAML 설정 로드와 검증. 학습을 GPU에 올리기 전에 잘못된 설정을 여기서 막는다."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from jeju_lora.prompts import INPUT_MODES

MODELS_YAML = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    license: str
    architecture: str
    card: str


def load_models(path: str | Path = MODELS_YAML) -> dict[str, ModelSpec]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    allowed = set(doc["allowed_licenses"])
    out = {}
    for key, m in doc["models"].items():
        if m["license"] not in allowed:
            raise ConfigError(f"{key}: 허용하지 않는 라이선스 {m['license']}")
        if len(str(m["revision"])) != 40:
            raise ConfigError(f"{key}: revision은 40자 커밋 해시로 고정한다")
        out[key] = ModelSpec(key=key, **m)
    return out


@dataclass
class LoraSection:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


@dataclass
class TrainSection:
    epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_batch_size: int = 32
    grad_accum: int = 1
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_length: int = 256
    max_train_samples: int | None = None
    max_eval_samples: int = 2000
    gradient_checkpointing: bool = True
    bf16: bool = True
    # T4처럼 bf16이 없는 GPU용. bf16과 함께 켤 수 없다
    fp16: bool = False
    logging_steps: int = 20
    eval_steps: int = 500
    save_steps: int = 500


@dataclass
class TrainConfig:
    model: str
    input_mode: str = "jeju"
    seed: int = 42
    output_dir: str = "outputs/lora"
    lora: LoraSection = field(default_factory=LoraSection)
    train: TrainSection = field(default_factory=TrainSection)


def _section(cls, raw: dict | None, name: str):
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{name}: 알 수 없는 키 {sorted(unknown)}")
    return cls(**raw)


def validate(cfg: TrainConfig, models: dict[str, ModelSpec]) -> None:
    if cfg.model not in models:
        raise ConfigError(f"model은 {sorted(models)} 중 하나: {cfg.model}")
    if cfg.input_mode not in INPUT_MODES:
        raise ConfigError(f"input_mode는 {INPUT_MODES} 중 하나: {cfg.input_mode}")
    lo, tr = cfg.lora, cfg.train
    if lo.r <= 0 or lo.alpha <= 0:
        raise ConfigError("lora.r, lora.alpha는 양수")
    if not 0 <= lo.dropout < 1:
        raise ConfigError("lora.dropout은 [0, 1)")
    if not (0 < tr.learning_rate < 1e-2):
        raise ConfigError("train.learning_rate가 범위를 벗어났다")
    if tr.epochs <= 0 or tr.per_device_batch_size <= 0 or tr.grad_accum <= 0:
        raise ConfigError("epochs·batch·grad_accum은 양수")
    if not 0 <= tr.warmup_ratio < 1:
        raise ConfigError("train.warmup_ratio는 [0, 1)")
    if tr.max_length < 64:
        raise ConfigError("train.max_length가 너무 짧다 (프롬프트만 60토큰 안팎)")
    if tr.max_train_samples is not None and tr.max_train_samples <= 0:
        raise ConfigError("train.max_train_samples는 양수 또는 null")
    if tr.bf16 and tr.fp16:
        raise ConfigError("train.bf16과 train.fp16은 함께 켤 수 없다")


def load_train_config(path: str | Path, models_path: str | Path = MODELS_YAML) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    top_known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - top_known
    if unknown:
        raise ConfigError(f"알 수 없는 키 {sorted(unknown)}")
    if "model" not in raw:
        raise ConfigError("model 키가 필요하다")
    cfg = TrainConfig(
        **{k: v for k, v in raw.items() if k not in {"lora", "train"}},
        lora=_section(LoraSection, raw.get("lora"), "lora"),
        train=_section(TrainSection, raw.get("train"), "train"),
    )
    validate(cfg, load_models(models_path))
    return cfg


def main() -> None:
    """셸에서 모델 정보를 꺼낼 때: python -m jeju_lora.config qwen3-4b hf_id"""
    import sys

    key, attr = sys.argv[1], sys.argv[2]
    print(getattr(load_models()[key], attr))


if __name__ == "__main__":
    main()
