from bedrock_fm import Titan, StreamDetails, CompletionDetails
from bedrock_fm.exceptions import BedrockExtraArgsError
from botocore.eventstream import EventStream
import pytest

fm = Titan.from_id('amazon.titan-text-express-v1')

def test_generate():
    r = fm.generate("test")
    assert type(r) is list

def test_generate_stream():
    r = fm.generate("test", stream=True)
    assert type(r) is EventStream
    assert r.__iter__ is not None
    c = 0
    for v in r:
        c += 1
        assert len(v) > 0
    assert c > 0

def test_generate_stream_details():
    r = fm.generate("test", stream=True, details=True)
    assert type(r) is StreamDetails
    assert r.stream.__iter__ is not None

def test_generate_details():
    r = fm.generate("test", details=True)
    assert type(r) is CompletionDetails
    assert r.body == '{"inputText": "test", "textGenerationConfig": {"maxTokenCount": 500, "stopSequences": [], "temperature": 0.7, "topP": 1}}'
    assert r.latency > 0
    assert r.prompt == "test"
    assert len(r.response) > 0

def test_extra_args():
    with pytest.raises(BedrockExtraArgsError):
        fm.generate("test", extra_args={"a":1})
    
