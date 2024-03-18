from bedrock_fm import Jurassic, Penalty
from bedrock_fm.exceptions import BedrockExtraArgsError
import pytest

fm = Jurassic.from_id('ai21.j2-mid-v1')

def test_args():
    b = fm.get_body("A", top_p=1, temperature=0.5, max_token_count=500, stop_sequences=[], extra_args={}, stream=False)
    assert b == '{"prompt": "A", "maxTokens": 500, "stopSequences": [], "temperature": 0.5, "topP": 1}'

def test_args_extra():
    b = fm.get_body("A", top_p=1, temperature=0.5, max_token_count=500, stop_sequences=[], extra_args={"countPenalty": {"scale": 1}}, stream=False)
    assert b == '{"countPenalty": {"scale": 1}, "prompt": "A", "maxTokens": 500, "stopSequences": [], "temperature": 0.5, "topP": 1}'


def test_args_unsupported():
    r = fm.generate("hello", count_penalty=Penalty(scale=1, apply_to_whitespaces=True))
    assert type(r) is list
    assert len(r[0]) > 0