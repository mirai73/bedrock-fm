from typing import List
import json
from .bedrock import BedrockEmbeddingsModel, EmbeddingType


class Embed(BedrockEmbeddingsModel):
    """Amazon Titan Embedding base class."""

    @classmethod
    def family(cls) -> str:
        return "cohere.embed"

    def get_body(self, data: List[str], type: EmbeddingType) -> str:
        return json.dumps(
            {
                "texts": data,
                "input_type": "search_document"
                if type == EmbeddingType.DOCUMENT
                else "search_query",
            }
        )

    def parse_response(self, response: bytes) -> List[List[float]]:
        response_body = json.loads(response.get("body").read())
        embeddings = response_body.get("embeddings")
        return embeddings
