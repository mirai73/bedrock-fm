from bedrock_fm import (
    MistralLarge,
    Model,
    CompletionDetails,
    Model,
    Human,
    System,
    Assistant,
)
from bedrock_fm.exceptions import BedrockExtraArgsError, BedrockInvocationError
import pytest

fm = MistralLarge.from_id(Model.MISTRAL_MISTRAL_LARGE_2402_V1_0)


def test_get_chat_prompt():
    p = fm.get_chat_prompt([System("S"), Human("H"), Assistant("A"), Human("H")])
    assert p == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "H"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "H"},
    ]


def test_generate():
    with pytest.raises(BedrockInvocationError):
        fm.generate("hello")


def test_chat():
    r = fm.chat(
        [System("You are a helpful assistant"), Human("What is your name?")], top_p=0.99
    )
    assert type(r) is list
    assert len(r) > 0
    assert len(r[0]) > 0
    assert isinstance(r[0], str)


def test_chat_stream():
    r = fm.chat(
        [System("You are a helpful assistant"), Human("What is your name?")],
        top_p=0.99,
        stream=True,
    )
    for x in r:
        assert type(x) is str
