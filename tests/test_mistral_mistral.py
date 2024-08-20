from bedrock_fm import (
    Mistral,
    Model,
    CompletionDetails,
    Model,
    Human,
    System,
    Assistant,
)
from bedrock_fm.exceptions import BedrockExtraArgsError, BedrockInvocationError
import pytest

fm = Mistral.from_id(Model.MISTRAL_MISTRAL_7B_INSTRUCT_V0_2)


def test_get_chat_prompt():
    p = fm.get_chat_prompt([System("S"), Human("H"), Assistant("A"), Human("H")])
    assert p == "[INST] <<SYS>>\nS\n<</SYS>>\n\nH [/INST] A </s><s>[INST] H [/INST]"


def test_generate():
    r = fm.generate("hello")
    assert type(r) is list
    assert len(r) == 1


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
