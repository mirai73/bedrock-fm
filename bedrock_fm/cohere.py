from .bedrock import (
    BedrockFoundationModel,
    CompletionDetails,
    StreamDetails,
    Assistant,
    BedrockFoundationModel,
    Human,
    System,
    MessageRole,
)
from .exceptions import BedrockExtraArgsError, BedrockInvocationError
import json
from typing import List, Any, Dict, Iterable, Optional, overload, Literal
from attrs import define


@define
class Command(BedrockFoundationModel):
    """Command is a text generation model for business use cases. Command is trained on data that supports reliable business applications,
    like text generation, summarization, copywriting, dialogue, extraction, and question answering.

    [Cohere Command](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html#model-parameters-cohere)
    models support the following arguments:

    "k": float,
    "return_likelihoods": "GENERATION|ALL|NONE",
    "num_generations": int

    """

    @classmethod
    def family(cls) -> str:
        return "cohere.command"

    def validate_extra_args(self, extra_args: Dict[str, Any]) -> bool:
        unsupp_args = []
        supported_args = ["return_likelihoods", "num_generations", "k"]
        for k in extra_args.keys():
            if k not in supported_args:
                unsupp_args.append(k)

        if len(unsupp_args) > 0:
            raise BedrockExtraArgsError(
                f"Arguments [{','.join(unsupp_args)}] are not supported by this model.\nOnly [${','.join(supported_args)}] are supported"
            )
        return True

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
                "prompt": prompt,
                "max_tokens": max_token_count,
                "temperature": temperature,
                "p": top_p,
                "stop_sequences": stop_sequences,
                "stream": stream,
            }
        )
        return json.dumps(body)

    @overload
    def generate(
        self,
        prompt: str,
        *,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: List[str],
        return_likelihoods: str,
        num_generations: int,
        k: int,
        details: Literal[True],
        stream: Literal[True],
    ) -> StreamDetails: ...

    @overload
    def generate(
        self,
        prompt: str,
        *,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: List[str],
        return_likelihoods: str,
        num_generations: int,
        k: int,
        details: Literal[True],
        stream: Literal[False],
    ) -> CompletionDetails: ...

    @overload
    def generate(
        self,
        prompt: str,
        *,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: List[str],
        return_likelihoods: str,
        num_generations: int,
        k: int,
        details: Literal[False],
        stream: Literal[True],
    ) -> Iterable[str]: ...

    @overload
    def generate(
        self,
        prompt: str,
        *,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: List[str],
        return_likelihoods: str,
        num_generations: int,
        k: int,
        details: Literal[False],
        stream: Literal[False],
    ) -> List[str]: ...

    def generate(
        self,
        prompt: str,
        *,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        max_token_count: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        return_likelihoods: Optional[str] = None,
        num_generations: Optional[int] = None,
        k: Optional[int] = None,
        details: Optional[Literal[False]] = False,
        stream: Optional[Literal[False]] = False,
    ) -> List[str] | Iterable[str] | CompletionDetails | StreamDetails:

        extra_args = {}
        if return_likelihoods:
            extra_args["return_likelihoods"] = return_likelihoods

        if num_generations:
            extra_args["num_generations"] = num_generations

        if k:
            extra_args["k"] = k

        return super().generate(
            prompt,
            top_p=top_p,
            temperature=temperature,
            max_token_count=max_token_count,
            stop_sequences=stop_sequences,
            extra_args=extra_args,
            details=details,
            stream=stream,
        )

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [self.get_text(r) for r in body["generations"]]

    def get_text(self, body: Dict[str, Any]) -> str:
        if not body.get("is_finished", False):
            t = body["text"]
            if t == "<EOS_TOKEN>":
                return "\n"
            else:
                return t
        else:
            return ""


class CommandR(BedrockFoundationModel):

    @classmethod
    def family(cls) -> str:
        return "cohere.command-r"

    def validate_extra_args(self, extra_args: Dict[str, Any]) -> bool:
        for k in list(extra_args.keys()):
            if k not in [
                "k",
                "documents",
                "search_queries_only",
                "preamble",
                "k",
                "prompt_truncation",
                "frequency_penalty",
                "presence_penalty",
                "seed",
                "return_prompt",
                "tools",
                "tool_results",
                "stop_sequences",
                "raw_prompting",
            ]:
                raise BedrockExtraArgsError(
                    f"Argument {k} not supported by Command-R models."
                )
        return True

    def get_chat_prompt(
        self, conversation: List[Human | Assistant | System]
    ) -> str | list:
        messages = []
        for c in conversation:
            if c.role == MessageRole.HUMAN:
                messages.append({"role": "USER", "message": c.content})
            elif c.role == MessageRole.ASSISTANT:
                messages.append({"role": "CHATBOT", "message": c.content})
        return messages

    def get_body(
        self,
        prompt: str | dict,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: List[str],
        extra_args: Dict[str, Any],
        stream: bool,
    ) -> str:
        body = extra_args.copy()
        if isinstance(prompt, list):
            body.update(extra_args)

            body.update(
                {
                    "message": prompt[-1]["message"],
                    "chat_history": prompt[:-1],
                    "max_tokens": max_token_count,
                    "p": top_p,
                    "temperature": temperature,
                }
            )
            return json.dumps(body)
        else:
            raise BedrockInvocationError(
                "Command-R models do not support generate api."
            )

    def get_text(self, body: Dict[str, Any]) -> str:
        return body["text"]

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [body["text"]]
