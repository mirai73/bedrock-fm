from bedrock_fm import Claude3, System, Human, Assistant, Model
import json
import pytest

fm = Claude3.from_id(Model.ANTHROPIC_CLAUDE_3_HAIKU_20240307_V1_0)


def test_conv_validation():
    p = fm.get_chat_prompt(
        [System("hello"), Human("bye"), Assistant("Ok"), Human("Nice")]
    )
    assert p == [
        {"content": "hello", "role": "system"},
        {"content": "bye", "role": "user"},
        {"content": "Ok", "role": "assistant"},
        {"content": "Nice", "role": "user"},
    ]


def test_conv_validation_2():
    with pytest.raises(ValueError):
        fm.chat([Human("bye"), System("hello"), Assistant("Ok")])


def test_conv_validation_3():
    with pytest.raises(ValueError):
        fm.chat([Assistant("Ok")])


def test_conv_validation_4():
    with pytest.raises(ValueError):
        fm.chat([System("Ok")])


def test_conv_validation_5():
    with pytest.raises(ValueError):
        fm.chat([Human("Ok"), Human("Hello")])


def test_chat():
    r = fm.chat([System("You are an helpful assistant"), Human("What is your name?")])
    assert type(r) == list
    assert type(r[0]) == str
    assert len(r[0]) > 0
