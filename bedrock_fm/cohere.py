from .bedrock import BedrockFoundationModel, CompletionDetails, StreamDetails
from .exceptions import BedrockExtraArgsError
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
    ) -> StreamDetails:
        ...

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
    ) -> CompletionDetails:
        ...

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
    ) -> Iterable[str]:
        ...

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
    ) -> List[str]:
        ...

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
        details: Optional[Literal[False]]=False,
        stream: Optional[Literal[False]]=False,
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
