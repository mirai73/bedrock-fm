from typing import List
import json
from .bedrock import BedrockEmbeddingsModel, EmbeddingType
from .exceptions import BedrockArgsError


class TitanEmbeddings(BedrockEmbeddingsModel):
    """Amazon Titan Embedding base class."""

    @classmethod
    def family(cls) -> str:
        return "amazon.titan-embed"

    def get_body(self, data: List[str], type: EmbeddingType) -> str:
        if len(data) != 1:
            raise BedrockArgsError(
                f"Titan embeddings do not support batch inference. Provide a single element array as input. {data}"
            )
        return json.dumps({"inputText": data[0]})

    def parse_response(self, response: bytes) -> List[List[float]]:
        response_body = json.loads(response.get("body").read())
        embedding = response_body.get("embedding")
        return [embedding]
