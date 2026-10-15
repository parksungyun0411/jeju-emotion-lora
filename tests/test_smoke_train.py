"""작은 랜덤 Llama로 학습 → 어댑터 저장 → 병합 → 생성 경로를 CPU에서 2스텝 돌린다.

코드 경로만 확인한다. 이 모델의 출력·손실은 의미가 없고 어디에도 보고하지 않는다.
torch·transformers·peft·trl·datasets가 없으면 건너뛴다(CI 기본 환경).
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from jeju_lora.config import LoraSection, TrainConfig, TrainSection  # noqa: E402
from jeju_lora.labels import INVALID, NUM_LABELS, parse_label  # noqa: E402
from jeju_lora.prompts import SYSTEM_PROMPT, build_messages  # noqa: E402

pytestmark = pytest.mark.smoke

TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
SPECIALS = ["<pad>", "<unk>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]


def _tiny_model(path, corpus):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=SPECIALS))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|end|>"
    )
    hf_tok.chat_template = TEMPLATE
    hf_tok.save_pretrained(path)
    cfg = LlamaConfig(
        vocab_size=len(hf_tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=hf_tok.pad_token_id,
        eos_token_id=hf_tok.eos_token_id,
    )
    LlamaForCausalLM(cfg).save_pretrained(path)


def test_train_merge_generate(tmp_path, items):
    from jeju_lora.data import write_jsonl
    from jeju_lora.generation import HFBackend
    from jeju_lora.merge import merge
    from jeju_lora.train_lora import build_datasets, make_trainer

    base = tmp_path / "tiny"
    corpus = [SYSTEM_PROMPT, "제주어 : 감정 :", *("중립 슬픔 행복 분노 놀람 공포 혐오".split())]
    corpus += [it["jeju"] for it in items]
    _tiny_model(base, corpus)

    write_jsonl(items, tmp_path / "train.jsonl")
    write_jsonl(items[:14], tmp_path / "val.jsonl")
    cfg = TrainConfig(
        model="qwen3-4b",
        output_dir=str(tmp_path / "out"),
        lora=LoraSection(r=4, alpha=8),
        train=TrainSection(
            per_device_batch_size=4,
            max_eval_samples=8,
            bf16=False,
            gradient_checkpointing=False,
            max_length=128,
            eval_steps=2,
            save_steps=1000,
        ),
    )
    train_ds, eval_ds = build_datasets(cfg, tmp_path / "train.jsonl", tmp_path / "val.jsonl")
    assert len(eval_ds) == 8
    trainer = make_trainer(cfg, str(base), None, train_ds, eval_ds, max_steps=2, use_cpu=True)
    result = trainer.train()
    assert result.global_step == 2
    adapter = tmp_path / "out" / "adapter"
    trainer.save_model(str(adapter))
    trainer.processing_class.save_pretrained(str(adapter))
    assert (adapter / "adapter_config.json").exists()

    merged = tmp_path / "merged"
    merge(str(base), None, str(adapter), str(merged))
    assert (merged / "config.json").exists()

    backend = HFBackend(str(merged), batch_size=4, max_new_tokens=3)
    outs = backend.generate([build_messages(it) for it in items[:5]])
    assert len(outs) == 5
    assert all(INVALID <= parse_label(o) < NUM_LABELS for o in outs)


def _tiny_setup(tmp_path, items):
    from jeju_lora.data import write_jsonl

    base = tmp_path / "tiny"
    corpus = [SYSTEM_PROMPT, "제주어 : 감정 :", *("중립 슬픔 행복 분노 놀람 공포 혐오".split())]
    corpus += [it["jeju"] for it in items]
    _tiny_model(base, corpus)
    write_jsonl(items, tmp_path / "train.jsonl")
    write_jsonl(items[:14], tmp_path / "val.jsonl")
    return base


def _tiny_cfg(tmp_path, **train):
    return TrainConfig(
        model="midm-2.0-mini",
        output_dir=str(tmp_path / "out"),
        lora=LoraSection(r=4, alpha=8),
        train=TrainSection(
            per_device_batch_size=4,
            max_eval_samples=8,
            bf16=False,
            gradient_checkpointing=False,
            eval_steps=2,
            save_steps=1000,
            **train,
        ),
    )


def test_truncated_answers_stop_training(tmp_path, items):
    """정답까지 잘린 예시를 TRL이 버리면 표본 수가 바뀌므로 학습 전에 멈춘다."""
    from jeju_lora.train_lora import build_datasets, make_trainer

    base = _tiny_setup(tmp_path, items)
    cfg = _tiny_cfg(tmp_path, max_length=8)
    train_ds, eval_ds = build_datasets(cfg, tmp_path / "train.jsonl", tmp_path / "val.jsonl")
    with pytest.raises(ValueError, match="max_length 8"):
        make_trainer(cfg, str(base), None, train_ds, eval_ds, max_steps=1, use_cpu=True)


def test_custom_chat_template_is_saved_with_adapter(tmp_path, items):
    from jeju_lora.train_lora import build_datasets, make_trainer

    base = _tiny_setup(tmp_path, items)
    template = tmp_path / "t.jinja"
    template.write_text(
        "{% for m in messages %}[{{ m.role }}]{{ m.content }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}[assistant]{% endif %}"
    )
    cfg = _tiny_cfg(tmp_path, max_length=128)
    cfg.chat_template = str(template)  # 절대 경로는 REPO_ROOT와 합쳐도 그대로다
    train_ds, eval_ds = build_datasets(cfg, tmp_path / "train.jsonl", tmp_path / "val.jsonl")
    trainer = make_trainer(cfg, str(base), None, train_ds, eval_ds, max_steps=1, use_cpu=True)
    trainer.train()
    adapter = tmp_path / "out" / "adapter"
    trainer.processing_class.save_pretrained(str(adapter))
    from transformers import AutoTokenizer

    assert AutoTokenizer.from_pretrained(str(adapter)).chat_template == template.read_text()
