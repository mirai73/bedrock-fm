from bedrock_fm import Claude3, CompletionDetails, Model, Human, System
from bedrock_fm.exceptions import BedrockExtraArgsError
import pytest

fm = Claude3.from_id(Model.ANTHROPIC_CLAUDE_3_HAIKU_20240307_V1_0)


def test_args():
    b = fm.get_body(
        prompt="A",
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={},
        stream=False,
    )
    assert (
        b
        == '{"messages": [{"role": "user", "content": "A"}], "max_tokens": 500, "stop_sequences": [], "temperature": 0.5, "top_p": 1, "anthropic_version": "bedrock-2023-05-31"}'
    )


def test_args_add_stop_seq():
    b = fm.get_body(
        prompt="A",
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=["Hello"],
        extra_args={},
        stream=False,
    )
    assert (
        b
        == '{"messages": [{"role": "user", "content": "A"}], "max_tokens": 500, "stop_sequences": ["Hello"], "temperature": 0.5, "top_p": 1, "anthropic_version": "bedrock-2023-05-31"}'
    )


def test_args_topk():
    b = fm.get_body(
        prompt="A",
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={"top_k": 200},
        stream=False,
    )
    assert (
        b
        == '{"top_k": 200, "messages": [{"role": "user", "content": "A"}], "max_tokens": 500, "stop_sequences": [], "temperature": 0.5, "top_p": 1, "anthropic_version": "bedrock-2023-05-31"}'
    )


def test_args_usnupported():
    with pytest.raises(BedrockExtraArgsError):
        fm.generate("Hello", extra_args={"top": 200})


def test_generate_details():
    r = fm.generate("test", details=True)
    assert type(r) is CompletionDetails
    assert (
        r.body
        == '{"messages": [{"role": "user", "content": "test"}], "max_tokens": 500, "stop_sequences": [], "temperature": 0.7, "top_p": 1, "anthropic_version": "bedrock-2023-05-31"}'
    )
    assert r.latency > 0
    assert r.prompt == "test"
    assert len(r.response) > 0


def test_chat():
    r = fm.chat([System("You are a helpful assistant"), Human("What is your name?")])
    assert type(r) is list
    assert len(r) > 0
    assert len(r[0]) > 0


def test_chat_streaming():
    r = fm.chat(
        [System("You are a helpful assistant"), Human("What is your name?")],
        stream=True,
    )
    for x in r:
        assert type(x) is str
