import boto3
import json
import time

from attrs import define, field, Factory
from .exceptions import BedrockArgsError
from typing import Any, List, Dict, Tuple
import logging
from PIL import Image
from .bedrock import Model
from abc import abstractmethod


@define
class BedrockImageModel:
    scale: float = field(default=0)
    steps: int = field(default=0)
    session: boto3.Session = field(
        default=Factory(lambda: boto3.Session()), kw_only=True
    )
    """A `boto3.Session` object to use to create an instance of the Bedrock client"""

    _client: Any = field(
        default=Factory(
            lambda self: self.session.client("bedrock-runtime"),
            takes_self=True,
        ),
        kw_only=True,
    )
    _model_id: str = field(default=None)

    @classmethod
    def from_id(cls, model_id: str | Model, **kwargs):
        if type(model_id) == Model:
            model_id = model_id.value
        if not cls._validate_model_id(model_id):
            raise BedrockArgsError(
                f"model_id {model_id} not compatible with by {cls.family}"
            )
        model = cls(**kwargs)
        model._model_id = model_id
        return model

    def _generate(
        self,
        prompts: List[Tuple],
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        **kwargs,
    ) -> List[Image.Image]:
        body = self.get_body(prompts, height, width, seed, **kwargs)
        resp = self._client.invoke_model(modelId=self._model_id, body=body)
        return self.get_images(resp)

    @abstractmethod
    def get_body(self, prompt: str, seed: int, extra_args: Dict[str, Any]) -> str: ...

    @abstractmethod
    def get_images(self, response: Dict[str, Any]) -> List[Image.Image]: ...
