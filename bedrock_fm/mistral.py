from .bedrock import Assistant, BedrockFoundationModel, Human, System, MessageRole
from .exceptions import BedrockExtraArgsError, BedrockInvocationError
import json
from typing import List, Any, Dict
from botocore.eventstream import EventStream
from attrs import define
from .meta import get_llama2_prompt


@define
class Mistral(BedrockFoundationModel):
    """Mistral and Mixtral models base class."""

    @classmethod
    def family(cls) -> str:
        return "mistral.mistral"

    def validate_extra_args(self, extra_args: Dict[str, Any]) -> bool:
        for k in list(extra_args.keys()):
            if k not in ["top_k"]:
                raise BedrockExtraArgsError(
                    f"Argument {k} not supported by Mistral models. Only top_k is supported"
                )
        return True

    def get_chat_prompt(
        self, conversation: List[Human | Assistant | System]
    ) -> str | list:
        return get_llama2_prompt(conversation)

    def get_body(
        self,
        prompt: str | list,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: List[str],
        extra_args: Dict[str, Any],
        stream: bool,
    ) -> str:
        body = extra_args.copy()
        if isinstance(prompt, str):
            if "top_k" in extra_args:
                body["top_k"] = extra_args["top_k"]

            body.update(
                {
                    "prompt": f"[INST] {prompt} [/INST] ",
                    "max_tokens": max_token_count,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop_sequences,
                }
            )
        else:
            body.update(
                {
                    "messages": prompt,
                    "max_tokens": max_token_count,
                    "top_p": top_p,
                    "temperature": temperature,
                }
            )
        return json.dumps(body)

    def get_text(self, body: Dict[str, Any]) -> str:
        return body["outputs"][0]["text"]

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [c["text"] for c in body["outputs"]]


class Mixtral(Mistral):

    @classmethod
    def family(cls) -> str:
        return "mistral.mixtral"


class MistralLarge(Mistral):

    @classmethod
    def family(cls) -> str:
        return "mistral.mistral-large"

    def get_body(
        self,
        prompt: str | list,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: List[str],
        extra_args: Dict[str, Any],
        stream: bool,
    ) -> str:
        body = extra_args.copy()
        if isinstance(prompt, str):
            raise BedrockInvocationError(
                "MistralLarge does not support the generate API"
            )

        body.update(
            {
                "messages": prompt,
                "max_tokens": max_token_count,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop_sequences,
            }
        )
        return json.dumps(body)

    def get_chat_prompt(
        self, conversation: List[Human | Assistant | System]
    ) -> str | list:

        messages = []
        for c in conversation:
            messages.append({"role": c.role, "content": c.content})
            if c.role == MessageRole.HUMAN:
                messages[-1]["role"] = "user"
        return messages

    def get_text(self, body: Dict[str, Any]) -> str:
        return body["choices"][0]["message"]["content"]

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [c["message"]["content"] for c in body["choices"]]
