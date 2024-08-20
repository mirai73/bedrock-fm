from bedrock_fm import Command, Model
from bedrock_fm.exceptions import BedrockExtraArgsError
import pytest

fm = Command.from_id(Model.COHERE_COMMAND_TEXT_V14)


def test_args():
    b = fm.get_body(
        "A",
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={},
        stream=False,
    )
    assert (
        b
        == '{"prompt": "A", "max_tokens": 500, "temperature": 0.5, "p": 1, "stop_sequences": [], "stream": false}'
    )


def test_args_extra():
    b = fm.get_body(
        "A",
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={"k": 1},
        stream=False,
    )
    assert (
        b
        == '{"k": 1, "prompt": "A", "max_tokens": 500, "temperature": 0.5, "p": 1, "stop_sequences": [], "stream": false}'
    )


def test_args():
    r = fm.generate("hello", k=2, num_generations=3)
    assert type(r) is list
    assert len(r) == 3
