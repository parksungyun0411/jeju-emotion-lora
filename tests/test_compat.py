"""kanana 설정 검증 우회. transformers가 없으면 건너뛴다(CI 기본 환경)."""

from pathlib import Path

import pytest


def test_registered_as_vllm_plugin():
    tomllib = pytest.importorskip("tomllib")
    doc = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    plugins = doc["project"]["entry-points"]["vllm.general_plugins"]
    assert plugins == {"jeju_lora_compat": "jeju_lora.compat:register"}


def test_kanana_shape_loads_only_with_explicit_head_dim():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig

    from jeju_lora import compat

    assert compat.apply()
    # kanana-1.5-2.1b와 같은 모양: 1792 / 24 heads, head_dim 128 명시
    c = LlamaConfig(hidden_size=1792, num_attention_heads=24, num_key_value_heads=8, head_dim=128)
    assert c.head_dim == 128
    with pytest.raises(Exception, match="multiple"):
        LlamaConfig(hidden_size=30, num_attention_heads=4)
