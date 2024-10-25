from bedrock_fm import Jamba, Model, Human, Assistant
from bedrock_fm.exceptions import BedrockExtraArgsError

fm = Jamba.from_id(Model.AI21_JAMBA_1_5_MINI_V1_0)


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
        == '{"messages": [{"role": "user", "content": "A"}], "max_tokens": 500, "stop": [], "temperature": 0.5, "stream": false}'
    )


def test_args_extra():
    b = fm.get_body(
        "A",
        top_p=1,
        temperature=0.5,
        max_token_count=500,
        stop_sequences=[],
        stream=False,
        extra_args={"n": 1},
    )
    assert (
        b
        == '{"n": 1, "messages": [{"role": "user", "content": "A"}], "max_tokens": 500, "stop": [], "temperature": 0.5, "stream": false}'
    )


def test_generate():
    b = fm.generate("Hello, how are you?")
    assert len(b) == 1


def test_chat_prompt():
    p = fm.get_chat_prompt(
        [Human("Hello, I am John"), Assistant("Hi John!"), Human("What's your name")]
    )
    assert p == [
        {"content": "Hello, I am John", "role": "user"},
        {"content": "Hi John!", "role": "assistant"},
        {"content": "What's your name", "role": "user"},
    ]


def test_chat():
    b = fm.chat(
        [Human("Hello, I am John"), Assistant("Hi John!"), Human("What's your name")],
        temperature=0,
    )
    assert (
        b[0].strip()
        == "My name is Jamba. I'm an AI assistant made by AI21, and I was trained in early 2024."
    )
