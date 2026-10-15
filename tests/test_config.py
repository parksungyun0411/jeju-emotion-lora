from pathlib import Path

import pytest
import yaml

from jeju_lora.config import MODELS_YAML, ConfigError, load_models, load_train_config

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("lora_*.yaml"))


def test_models_yaml_only_permissive_licences():
    models = load_models()
    assert set(models) == {"kanana-1.5-2.1b", "qwen3-4b", "midm-2.0-mini"}
    assert {m.license for m in models.values()} <= {"apache-2.0", "mit"}
    assert not any("EXAONE" in m.hf_id for m in models.values())


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_validate(path):
    cfg = load_train_config(path)
    assert path.stem == f"lora_{cfg.model}"
    assert cfg.seed == 42


def _write(tmp_path, doc) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True))
    return p


@pytest.mark.parametrize(
    "doc,match",
    [
        ({"input_mode": "jeju"}, "model 키"),
        ({"model": "exaone-4.0-1.2b"}, "model은"),
        ({"model": "qwen3-4b", "lr": 1e-4}, "알 수 없는 키"),
        ({"model": "qwen3-4b", "lora": {"rank": 8}}, "lora: 알 수 없는 키"),
        ({"model": "qwen3-4b", "lora": {"r": 0}}, "양수"),
        ({"model": "qwen3-4b", "lora": {"dropout": 1.0}}, "dropout"),
        ({"model": "qwen3-4b", "train": {"learning_rate": 0.5}}, "learning_rate"),
        ({"model": "qwen3-4b", "train": {"max_length": 32}}, "max_length"),
        ({"model": "qwen3-4b", "input_mode": "std"}, "input_mode"),
        ({"model": "qwen3-4b", "train": {"max_train_samples": 0}}, "max_train_samples"),
    ],
)
def test_invalid_configs(tmp_path, doc, match):
    with pytest.raises(ConfigError, match=match):
        load_train_config(_write(tmp_path, doc))


def test_rejects_non_permissive_licence(tmp_path):
    doc = yaml.safe_load(MODELS_YAML.read_text())
    doc["models"]["qwen3-4b"]["license"] = "other"
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ConfigError, match="라이선스"):
        load_models(p)


def test_rejects_unpinned_revision(tmp_path):
    doc = yaml.safe_load(MODELS_YAML.read_text())
    doc["models"]["qwen3-4b"]["revision"] = "main"
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ConfigError, match="revision"):
        load_models(p)


T4_CONFIGS = sorted((Path(__file__).parents[1] / "configs" / "t4").glob("lora_*.yaml"))


@pytest.mark.parametrize("path", T4_CONFIGS, ids=lambda p: p.name)
def test_t4_configs_differ_only_in_precision_and_sample_cap(path):
    """T4 설정은 24GB 설정에서 정밀도(fp16)와 학습 표본 상한만 바꾼다."""
    t4 = load_train_config(path)
    base = load_train_config(path.parents[1] / path.name)
    assert t4.train.fp16 and not t4.train.bf16
    changed = {f for f in vars(t4.train) if getattr(t4.train, f) != getattr(base.train, f)} - {
        "fp16",
        "bf16",
        "max_train_samples",
    }
    assert changed == set()
    assert (t4.model, t4.seed, t4.lora, t4.output_dir) == (
        base.model,
        base.seed,
        base.lora,
        base.output_dir,
    )


def test_t4_configs_exist_for_every_model():
    assert {p.stem.removeprefix("lora_") for p in T4_CONFIGS} == set(load_models())
