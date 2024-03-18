from bedrock_fm import Embed, Model, EmbeddingType
import json

emv1 = Embed.from_id(Model.COHERE_EMBED_ENGLISH_V3)

def test_embeddings_body():
    b = emv1.get_body(["Hello, how are you"],  type=EmbeddingType.DOCUMENT )
    assert  json.loads(b) == {"texts": ["Hello, how are you"], "input_type": "search_document"}

def test_doc_embeddings():
    b = emv1.generate_for_documents(["Hello, how are you"])
    assert len(b[0]) == 1024

def test_query_embeddings():
    b = emv1.generate_for_query("Hello, how are you")
    assert len(b) == 1024