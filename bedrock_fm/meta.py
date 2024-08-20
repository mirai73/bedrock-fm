from .bedrock import Assistant, BedrockFoundationModel, Human, System
from .exceptions import BedrockExtraArgsError, BedrockInvocationError
import json
from typing import List, Any, Dict
from botocore.eventstream import EventStream
from attrs import define
from .bedrock import MessageRole


def get_llama2_prompt(conversation: List[Human | Assistant | System]) -> str:
    prompts = []
    if conversation[0].role == MessageRole.SYSTEM:
        prompts.append(f"<<SYS>>\n{conversation[0].content}\n<</SYS>>\n\n")
        conversation = conversation[1:]
    prompts.append(f"{conversation[0].content}")  # Human
    conversation = conversation[1:]
    if len(conversation) > 0:
        prompts.append(f" [/INST] {conversation[0].content} </s>")  # Assistant
        conversation = conversation[1:]
        for m in zip(conversation[:-1][0::2], conversation[1::2]):
            prompts.append(
                f"<s>[INST] {m[0].content} [/INST] {m[1].content} </s>"
            )  # Human -> Assistant
        prompts.append(f"<s>[INST] {conversation[-1].content}")  # Human

    return f'[INST] {"".join(prompts)} [/INST]'


def get_llama3_prompt(conversation: List[Human | Assistant | System]) -> str:
    SYSTEM_MSG = "<|start_header_id|>system<|end_header_id|>\n\n{msg}<|eot_id|>"
    USER_MSG = "<|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|>\n"
    AI_MSG = "<|start_header_id|>assistant<|end_header_id|>\n\n{msg}<|eot_id|>\n"
    prompts = []
    if conversation[0].role == MessageRole.SYSTEM:
        prompts.append(SYSTEM_MSG.format(msg=conversation[0].content))
        conversation = conversation[1:]
    prompts.append(USER_MSG.format(msg=conversation[0].content))  # Human

    conversation = conversation[1:]
    if len(conversation) > 0:
        prompts.append(AI_MSG.format(msg=conversation[0].content))  # Assistant
        conversation = conversation[1:]
        for m in zip(conversation[:-1][0::2], conversation[1::2]):
            prompts.append(USER_MSG.format(msg=m[0].content))
            prompts.append(AI_MSG.format(msg=m[1].content))  # Human -> Assistant
        prompts.append(USER_MSG.format(msg=conversation[-1].content))  # Human

    return f'<|begin_of_text|>{"".join(prompts)}<|start_header_id|>assistant<|end_header_id|>'


@define
class Llama2Chat(BedrockFoundationModel):
    """A dialogue use case optimized variant of Llama 2 models.
    Llama 2 is an auto-regressive language model that uses an optimized transformer architecture.
    Llama 2 is intended for commercial and research use in English.

    [Llama2 Chat](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html) does
    not support any `extra_args`.
    """

    @classmethod
    def family(cls) -> str:
        return "meta.llama2"

    def validate_extra_args(self, extra_args: Dict[str, Any]) -> bool:
        if len(extra_args.keys()) > 0:
            raise BedrockExtraArgsError("Llama2 Chat does not support any extra args")

    def get_chat_prompt(
        self, conversation: List[Human | Assistant | System]
    ) -> str | list:
        return conversation

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
        if isinstance(prompt, str):
            raise BedrockInvocationError(
                "Llama2Chat model does not support generate api"
            )
        body = extra_args.copy()
        body.update(
            {
                "prompt": get_llama2_prompt(prompt),
                "max_gen_len": max_token_count,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return json.dumps(body)

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [body["generation"][2:]]

    def get_text(self, body):
        return body["generation"]


@define
class Llama3Instruct(BedrockFoundationModel):
    """A dialogue use case optimized variant of Llama 3 models.
    Llama 3 is an auto-regressive language model that uses an optimized transformer architecture.
    Llama 3 is intended for commercial and research use in English.

    [Llama3 Chat](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html) does
    not support any `extra_args`.
    """

    @classmethod
    def family(cls) -> str:
        return "meta.llama3"

    def get_chat_prompt(
        self, conversation: List[Human | Assistant | System]
    ) -> str | list:
        return conversation

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
        if isinstance(prompt, str):
            raise BedrockInvocationError(
                "Llama3Instruct model does not support generate api"
            )
        body = extra_args.copy()
        body.update(
            {
                "prompt": get_llama3_prompt(prompt),
                "max_gen_len": max_token_count,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return json.dumps(body)

    def get_text(self, body):
        return body["generation"]

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [body["generation"][2:]]
