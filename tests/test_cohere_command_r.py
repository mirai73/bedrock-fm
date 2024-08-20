from bedrock_fm import (
    CommandR,
    Model,
    CompletionDetails,
    Model,
    Human,
    System,
    Assistant,
)
from bedrock_fm.exceptions import BedrockExtraArgsError, BedrockInvocationError
import pytest

fm = CommandR.from_id(Model.COHERE_COMMAND_R_V1_0)


def test_args():
    with pytest.raises(BedrockInvocationError):
        fm.get_body(
            "A",
            top_p=1,
            temperature=0.5,
            max_token_count=500,
            stop_sequences=[],
            extra_args={},
            stream=False,
        )


def test_get_chat_prompt():
    p = fm.get_chat_prompt([System("S"), Human("H"), Assistant("A")])
    assert p == [{"role": "USER", "message": "H"}, {"role": "CHATBOT", "message": "A"}]


def test_args_extra():
    b = fm.get_body(
        [{"role": "USER", "message": "H"}],
        top_p=0.99,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={
            "k": 0.99,
            "documents": [{"title": "T", "snippet": "S"}],
            "search_queries_only": True,
            "preamble": "P",
            "prompt_truncation": "P",
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "seed": 1,
            "return_prompt": True,
            "tools": [
                {
                    "name": "N",
                    "description": "D",
                    "parameter_definitions": {
                        "parameter name": {
                            "description": "D",
                            "type": "T",
                            "required": True,
                        }
                    },
                }
            ],
            "tool_results": [
                {
                    "call": {"name": "N", "parameters": {"parameter name": "N"}},
                    "outputs": [{"text": "T"}],
                }
            ],
            "stop_sequences": ["S"],
            "raw_prompting": False,
        },
        stream=False,
    )
    assert (
        b
        == '{"k": 0.99, "documents": [{"title": "T", "snippet": "S"}], "search_queries_only": true, "preamble": "P", "prompt_truncation": "P", "frequency_penalty": 0, "presence_penalty": 0, "seed": 1, "return_prompt": true, "tools": [{"name": "N", "description": "D", "parameter_definitions": {"parameter name": {"description": "D", "type": "T", "required": true}}}], "tool_results": [{"call": {"name": "N", "parameters": {"parameter name": "N"}}, "outputs": [{"text": "T"}]}], "stop_sequences": ["S"], "raw_prompting": false, "message": "H", "chat_history": [], "max_tokens": 500, "p": 0.99, "temperature": 0.5}'
    )


def test_args():
    with pytest.raises(BedrockInvocationError):
        fm.generate("hello")


def test_chat():
    r = fm.chat(
        [System("You are a helpful assistant"), Human("What is your name?")], top_p=0.99
    )
    assert type(r) is list
    assert len(r) > 0
    assert len(r[0]) > 0
