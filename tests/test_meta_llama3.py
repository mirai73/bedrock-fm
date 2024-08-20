from bedrock_fm import (
    Llama3Instruct,
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

fm = Llama3Instruct.from_id(Model.META_LLAMA3_8B_INSTRUCT_V1_0)


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
    b = json.loads(b)
    assert b == {
        "prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nA<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>",
        "max_gen_len": 500,
        "temperature": 0.5,
        "top_p": 1,
    }


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
    assert (
        p
        == """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

H<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

A<|eot_id|>
<|start_header_id|>user<|end_header_id|>

H<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    )


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
    assert (
        p
        == """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

S<|eot_id|><|start_header_id|>user<|end_header_id|>

H<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

A<|eot_id|>
<|start_header_id|>user<|end_header_id|>

H<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    )


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
    assert (
        p
        == """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

S<|eot_id|><|start_header_id|>user<|end_header_id|>

H<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    )


def test_get_chat_prompt_4():
    p = fm.get_body(
        [Human("H")],
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        extra_args={},
        stream=False,
    )
    p = json.loads(p)["prompt"]
    assert (
        p
        == """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

H<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    )


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
