"""transformers 5.x의 LlamaConfig 검증 우회 (ko-judge-lora의 compat.py와 같은 패치).

transformers 5.17·5.18의 LlamaConfig.validate_architecture는 hidden_size가 num_attention_heads로
나눠떨어지지 않으면 설정을 거부한다. kanana-1.5-2.1b는 hidden_size 1792, head 24에
head_dim 128을 명시하므로 실제 구조는 문제가 없는데도 불러오지 못한다.
head_dim이 명시된 경우에만 검사를 건너뛴다. 학습·병합·양자화·HF 생성 프로세스는 `apply()`를 부르고,
vLLM은 `vllm.general_plugins` 진입점(`register`)으로 같은 패치를 건다.
"""

from __future__ import annotations

_APPLIED = False


def apply() -> bool:
    """패치를 적용했으면 True. transformers가 없거나 검증 함수가 없으면 False."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from transformers.models.llama import configuration_llama as cl
    except ImportError:
        return False
    original = getattr(cl.LlamaConfig, "validate_architecture", None)
    if original is None:
        return False

    def validate_architecture(self):
        # __post_init__이 head_dim을 hidden_size // heads로 채우므로, 그 값과 다르면 명시된 것이다
        head_dim = getattr(self, "head_dim", None)
        if head_dim and head_dim != self.hidden_size // self.num_attention_heads:
            return None
        return original(self)

    cl.LlamaConfig.validate_architecture = validate_architecture
    # huggingface_hub @strict는 클래스 생성 시점에 검증 함수 목록을 따로 저장한다
    validators = getattr(cl.LlamaConfig, "__class_validators__", None)
    if validators is not None:
        cl.LlamaConfig.__class_validators__ = [
            validate_architecture if v is original else v for v in validators
        ]
    _APPLIED = True
    return True


def register() -> None:
    """vLLM general plugin 진입점."""
    apply()
