from bedrock_fm import (
    Llama2Chat,
    Model,
    CompletionDetails,
    Model,
    Human,
    System,
    Assistant,
)
from bedrock_fm.exceptions import BedrockExtraArgsError, BedrockInvocationError
import pytest
import json

fm = Llama2Chat.from_id(Model.META_LLAMA2_13B_CHAT_V1)


def test_args():

    b = fm.get_body(
        [Human("A")],
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={},
        stream=False,
    )
    assert (
        b
        == '{"prompt": "[INST] A [/INST]", "max_gen_len": 500, "temperature": 0.5, "top_p": 1}'
    )


def test_get_chat_prompt():
    p = fm.get_body(
        [Human("H"), Assistant("A"), Human("H")],
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={},
        stream=False,
    )
    p = json.loads(p)["prompt"]
    assert p == "[INST] H [/INST] A </s><s>[INST] H [/INST]"


def test_get_chat_prompt_2():
    p = fm.get_body(
        [System("S"), Human("H"), Assistant("A"), Human("H")],
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={},
        stream=False,
    )
    p = json.loads(p)["prompt"]
    assert p == "[INST] <<SYS>>\nS\n<</SYS>>\n\nH [/INST] A </s><s>[INST] H [/INST]"


def test_get_chat_prompt_3():
    p = fm.get_body(
        [System("S"), Human("H")],
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={},
        stream=False,
    )
    p = json.loads(p)["prompt"]
    assert p == "[INST] <<SYS>>\nS\n<</SYS>>\n\nH [/INST]"


def test_chat():
    r = fm.chat(
        [System("You are a helpful assistant"), Human("What is your name?")], top_p=0.99
    )
    assert type(r) is list
    assert len(r) > 0
    assert len(r[0]) > 0


def test_generate():
    with pytest.raises(BedrockInvocationError):
        fm.generate("hello")


def test_chat_streaming():
    r = fm.chat(
        [System("You are a helpful assistant"), Human("What is your name?")],
        stream=True,
    )
    for x in r:
        assert type(x) is str
