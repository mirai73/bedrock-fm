from .bedrock import Assistant, BedrockFoundationModel, Human, System
from .exceptions import BedrockExtraArgsError
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
    prompts.append(f"{conversation[0].content}")
    conversation = conversation[1:]
    if len(conversation) == 0:
        return "".join(prompts)
    prompts.append(f" [/INST] {conversation[0].content} </s>")
    conversation = conversation[1:]
    for m in zip(conversation[:-1][0::2], conversation[1::2]):
        prompts.append(f"<s>[INST] {m[0].content} [/INST] {m[1].content} </s>")
    prompts.append(f"<s>[INST] {conversation[-1].content}")

    return "".join(prompts)


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
        return "meta.llama2-chat"

    def validate_extra_args(self, extra_args: Dict[str, Any]) -> bool:
        if len(extra_args.keys()) > 0:
            raise BedrockExtraArgsError("Llama2 Chat does not support any extra args")

    def get_chat_prompt(
        self, conversation: List[Human | Assistant | System]
    ) -> str | list:
        return get_llama2_prompt(conversation)

    def get_body(
        self,
        prompt: str,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: List[str],
        extra_args: Dict[str, Any],
        stream: bool,
    ) -> str:
        body = extra_args.copy()
        body.update(
            {
                "prompt": f"[INST] {prompt} [/INST] ",
                "max_gen_len": max_token_count,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return json.dumps(body)

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [body["generation"]]
