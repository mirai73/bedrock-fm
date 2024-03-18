from bedrock_fm import Claude, CompletionDetails
from bedrock_fm.exceptions import BedrockExtraArgsError
import pytest

fm = Claude.from_id('anthropic.claude-instant-v1')

def test_args():
    b = fm.get_body(prompt="A", top_p=1, temperature=0.5, max_token_count=500, stop_sequences=[], extra_args={}, stream=False)
    assert b == '{"prompt": "Human: A\\n\\nAssistant:", "max_tokens_to_sample": 500, "stop_sequences": ["\\n\\nHuman:"], "temperature": 0.5, "top_p": 1, "anthropic_version": "bedrock-2023-05-31"}'
    

def test_args_add_stop_seq():
    b = fm.get_body(prompt="A", top_p=1, temperature=0.5, max_token_count=500, stop_sequences=["Hello"], extra_args={}, stream=False)
    assert b == '{"prompt": "Human: A\\n\\nAssistant:", "max_tokens_to_sample": 500, "stop_sequences": ["Hello", "\\n\\nHuman:"], "temperature": 0.5, "top_p": 1, "anthropic_version": "bedrock-2023-05-31"}'
    
def test_args_topk():
    b = fm.get_body(prompt="A", top_p=1, temperature=0.5, max_token_count=500, stop_sequences=[], extra_args={"top_k": 200}, stream=False)
    assert b == '{"top_k": 200, "prompt": "Human: A\\n\\nAssistant:", "max_tokens_to_sample": 500, "stop_sequences": ["\\n\\nHuman:"], "temperature": 0.5, "top_p": 1, "anthropic_version": "bedrock-2023-05-31"}'

def test_args_usnupported():
    with pytest.raises(BedrockExtraArgsError):
        fm.generate("Hello", extra_args={"top": 200})

def test_generate_details():
    r = fm.generate("test", details=True)
    assert type(r) is CompletionDetails
    assert r.body == '{"prompt": "Human: test\\n\\nAssistant:", "max_tokens_to_sample": 500, "stop_sequences": ["\\n\\nHuman:"], "temperature": 0.7, "top_p": 1, "anthropic_version": "bedrock-2023-05-31"}'
    assert r.latency > 0
    assert r.prompt == "test"
    assert len(r.response) > 0