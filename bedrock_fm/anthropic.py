from .bedrock import Assistant, BedrockFoundationModel, Human, System, MessageRole
from .exceptions import BedrockExtraArgsError
import json
from attrs import define
from typing import List, Any, Dict
import io
from base64 import b64encode

HUMAN_PROMPT = "\n\nHuman:"
ASSISTANT_PROMPT = "\n\nAssistant:"


@define
class Claude(BedrockFoundationModel):
    """Anthropic offers the Claude family of large language models purpose built for conversations, summarization,
    Q&A, workflow automation, coding and more. Early customers report that Claude is much less likely to produce harmful
    outputs, easier to converse with, and more steerable - so you can get your desired output with less effort.
    Claude can also take direction on personality, tone, and behavior.

    [Anthropic Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html#model-parameters-claude)
    models support the following `extra_args`

        {
            "top_k": int
        }
    """

    @classmethod
    def family(cls) -> str:
        return "anthropic.claude"

    def validate_extra_args(self, extra_args: Dict[str, Any]) -> None:
        unsupp_args = []
        for k in extra_args.keys():
            if k not in ["top_k"]:
                unsupp_args.append(k)

        if len(unsupp_args) > 0:
            raise BedrockExtraArgsError(
                f"Arguments [{','.join(unsupp_args)}] are not supported by this model"
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
        s = list(stop_sequences)
        if HUMAN_PROMPT not in s:
            s.append(HUMAN_PROMPT)

        claude_prompt = prompt
        if not claude_prompt.startswith(HUMAN_PROMPT):
            claude_prompt = HUMAN_PROMPT + claude_prompt

        a_i = -1
        while True:
            try:
                a_i = claude_prompt.index(ASSISTANT_PROMPT, a_i + 1)
            except:
                break

        if a_i < 0 or (a_i > 0 and "Human:" in claude_prompt[a_i:]):
            claude_prompt = claude_prompt + ASSISTANT_PROMPT

        body.update(
            {
                "prompt": claude_prompt,
                "max_tokens_to_sample": max_token_count,
                "stop_sequences": s,
                "temperature": temperature,
                "top_p": top_p,
                "anthropic_version": "bedrock-2023-05-31",
            }
        )
        return json.dumps(body)

    def get_chat_prompt(self, conversation: List[Human | Assistant | System]) -> str:
        prompts = []
        if conversation[0].role == MessageRole.SYSTEM:
            prompts.append(
                f"{HUMAN_PROMPT} {conversation[0].content} {conversation[1].content}"
            )
            conversation = conversation[2:]
        if len(conversation) == 0:
            return "".join(prompts)

        for m in conversation:
            role = HUMAN_PROMPT if m.role == MessageRole.HUMAN else ASSISTANT_PROMPT
            prompts.append(f"{role} {m.content}")

        return "".join(prompts)

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [self.get_text(body)]

    def get_text(self, body: Dict[str, Any]) -> str:
        return body["completion"]


@define
class Claude3(BedrockFoundationModel):
    """Anthropic offers the Claude family of large language models purpose built for conversations, summarization,
    Q&A, workflow automation, coding and more. Early customers report that Claude is much less likely to produce harmful
    outputs, easier to converse with, and more steerable - so you can get your desired output with less effort.
    Claude can also take direction on personality, tone, and behavior.

    [Anthropic Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html#model-parameters-claude)
    models support the following `extra_args`

        {
            "top_k": int
        }
    """

    @classmethod
    def family(cls) -> str:
        return "anthropic.claude-3"

    def validate_extra_args(self, extra_args: Dict[str, Any]) -> None:
        unsupp_args = []
        for k in extra_args.keys():
            if k not in ["top_k"]:
                unsupp_args.append(k)

        if len(unsupp_args) > 0:
            raise BedrockExtraArgsError(
                f"Arguments [{','.join(unsupp_args)}] are not supported by this model"
            )

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
        if type(prompt) == str:
            prompt = [{"role": "user", "content": prompt}]
        else:
            if prompt[0].get("role") == "system":
                body.update({"system": prompt[0].get("content", "")})
                prompt = prompt[1:]

        s = list(stop_sequences)
        body.update(
            {
                "messages": prompt,
                "max_tokens": max_token_count,
                "stop_sequences": s,
                "temperature": temperature,
                "top_p": top_p,
                "anthropic_version": "bedrock-2023-05-31",
            }
        )
        return json.dumps(body)

    def get_chat_prompt(self, conversation: List[Human | Assistant | System]) -> list:
        prompts = []
        if conversation[0].role == MessageRole.SYSTEM:
            # ignore SYSTEM messages here, would need to use a specific "system" parameter
            prompts.append({"role": "system", "content": conversation[0].content})
            conversation = conversation[1:]
        if len(conversation) == 0:
            return prompts

        for m in conversation:
            if m.role == MessageRole.HUMAN:
                if len(m.images) > 0:
                    user_msg = [{"type": "text", "text": m.content}]
                    for img in m.images:
                        out = io.BytesIO()
                        img.save(out, format="PNG")
                        user_msg.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "data": str(b64encode(out.getvalue()), "ascii"),
                                    "media_type": "image/png",
                                },
                            }
                        )
                    prompts.append({"role": "user", "content": user_msg})
                else:
                    prompts.append({"role": "user", "content": m.content})
            else:
                prompts.append({"role": "assistant", "content": m.content})
        return prompts

    def process_response_body(self, body: Dict[str, Any]) -> List[str]:
        return [self.get_text(body)]

    def get_text(self, body: Dict[str, Any]) -> str:
        if body["type"] == "content_block_delta":
            return body["delta"]["text"]
        elif body["type"] == "message":
            return body["content"][0]["text"]
        else:
            return ""
