from bedrock_fm import  TitanEmbeddings

emv1 = TitanEmbeddings.from_id("amazon.titan-embed-text-v1")

def test_embeddings_body():
    b = emv1.get_body("Hello, how are you")
    assert b == '{"inputText": "Hello, how are you"}'

def test_embeddings():
    b = emv1.generate("Hello, how are you")
    assert len(b) == 1536
