from .bedrock import BedrockFoundationModel, CompletionDetails, StreamDetails
from .exceptions import BedrockExtraArgsError
import json
from typing import List, Any, Dict, Optional, overload, Literal
from botocore.eventstream import EventStream
from attrs import define, field
from typing import Iterable


@define
class Penalty:
    """Penalty object"""

    scale: int
    apply_to_whitespaces:bool = field(default=False)
    apply_to_punctuations: bool = field(default=False)
    apply_to_numbers: bool = field(default=False)
    apply_to_stopwords: bool = field(default=False)
    apply_to_emojis: bool = field(default=False)

    def to_dict(self):
        return {
            "scale": self.scale,
            "applyToWhitespaces": self.apply_to_whitespaces,
            "applyToPunctuations": self.apply_to_punctuations,
            "applyToNumbers": self.apply_to_numbers,
            "applyToStopwords": self.apply_to_stopwords,
            "applyToEmojis": self.apply_to_emojis,
        }


@define
class Jurassic(BedrockFoundationModel):
    """AI21 offers Jurassic-2, state-of-the-art LLMs that enable developers and businesses to build their
    own generative AI-driven applications and services. Jurassic-2 models power language generation and comprehension features
    in thousands of live applications, including long and short-form text generation, contextual question answering, creative writing,
    summarization, and classification. The models are designed to follow natural language instructions and context without
    requiring examples (zero-shot) and are trained on a massive corpus of web text with recent data updated to mid-2022.
    In addition to English, Jurassic-2 supports six other languages: Spanish, French, German, Italian, Portuguese, and Dutch.

    [Jurassic-2](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html#model-parameters-jurassic2) models support the following `extra_args`:


        {
            "countPenalty": {
                "scale": int [0,5],
                "applyToWhitespaces": boolean,
                "applyToPunctuations": boolean,
                "applyToNumbers": boolean,
                "applyToStopwords": boolean,
                "applyToEmojis": boolean
            },
            "presencePenalty": {
                "scale": float [0,1],
                "applyToWhitespaces": boolean,
                "applyToPunctuations": boolean,
                "applyToNumbers": boolean,
                "applyToStopwords": boolean,
                "applyToEmojis": boolean

            },
            "frequencyPenalty": {
                "scale": int [0,500],
                "applyToWhitespaces": boolean,
                "applyToPunctuations": boolean,
                "applyToNumbers": boolean,
                "applyToStopwords": boolean,
                "applyToEmojis": boolean

            }
        }
    """

    @define
    class CountPenalty(Penalty):
        """Count Penalty object"""

        pass

    @define
    class PresencePenalty(Penalty):
        """Presence Penalty object"""

        pass

    @define
    class FrequencyPenalty(Penalty):
        """Frequency Penalty object"""

        pass

    @classmethod
    def family(cls) -> str:
        return "ai21.j2"

    def validate_extra_args(self, extra_args: Dict[str, Any]):
        unsupp_args = []
        for k in extra_args.keys():
            if k not in ["countPenalty", "presencePenalty", "frequencePenalty"]:
                unsupp_args.append(k)

        if len(unsupp_args) > 0:
            raise BedrockExtraArgsError(
                f"Arguments [{','.join(unsupp_args)}] are not supported by this model"
            )

    @overload
    def generate(
        self,
        prompt: str,
        *,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: List[str],
        count_penalty: CountPenalty,
        frequency_penalty: FrequencyPenalty,
        presence_penalty: PresencePenalty,
        details: Literal[False],
        stream: Literal[False],
    ) -> List[str]:
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
        count_penalty: CountPenalty,
        frequency_penalty: FrequencyPenalty,
        presence_penalty: PresencePenalty,
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
        count_penalty: CountPenalty,
        frequency_penalty: FrequencyPenalty,
        presence_penalty: PresencePenalty,
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
        count_penalty: CountPenalty,
        frequency_penalty: FrequencyPenalty,
        presence_penalty: PresencePenalty,
        details: Literal[True],
        stream: Literal[True],
    ) -> StreamDetails:
        ...

    def generate(
        self,
        prompt: str,
        *,
        top_p: float = 0.9,
        temperature: float = 0.5,
        max_token_count: int = 500,
        stop_sequences: List[str] = [],
        count_penalty: Optional[CountPenalty] = None,
        frequency_penalty: Optional[FrequencyPenalty] = None,
        presence_penalty: Optional[PresencePenalty] = None,
        details: bool = False,
        stream: bool = False,
    ) -> List[str] | CompletionDetails | StreamDetails | Iterable[str]:
        extra_args = {}
        if count_penalty is not None:
            extra_args["countPenalty"] = count_penalty.to_dict()
        if frequency_penalty is not None:
            extra_args["frequencyPenalty"] = frequency_penalty.to_dict()
        if presence_penalty is not None:
            extra_args["presencePenalty"] = presence_penalty.to_dict()

        return super().generate(
            prompt,
            top_p=top_p,
            temperature=temperature,
            max_token_count=max_token_count,
            stop_sequences=stop_sequences,
            extra_args=extra_args,
            stream=stream,
            details=details,
        )

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
                "maxTokens": max_token_count,
                "stopSequences": stop_sequences,
                "temperature": temperature,
                "topP": top_p,
            }
        )
        return json.dumps(body)

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [self.get_text(r) for r in body["completions"]]

    def get_text(self, body: Dict[str, Any]) -> str:
        return body["data"]["text"]
