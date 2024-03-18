from typing import Any, Dict, List, Tuple, Optional, overload

from bedrock_fm.bedrock import Model
from .bedrock_image import BedrockImageModel
from .exceptions import BedrockExtraArgsError
import json
from attrs import define, asdict
from enum import Enum
from io import BytesIO
from PIL import Image
from base64 import b64decode

resolutions = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
]


class SDStylePresets(Enum):
    THREE_D_MODEL = "3d-model"
    ANALOG_FILM = "analog-film"
    ANIME = "anime"
    CINEMATIC = "cinematic"
    COMIC_BOOK = "comic-book"
    DIGITAL_ART = "digital-art"
    ENHANCE = "enhance"
    FANTASY_ART = "fantasy-art"
    ISOMETRIC = "isometric"
    LINE_ART = "line-art"
    LOW_POLY = "low-poly"


@define
class SDXL(BedrockImageModel):
    """ """

    @classmethod
    def from_id(cls, model_id: str | Model, **kwargs):
        return super().from_id(model_id, **kwargs)

    @classmethod
    def _validate_model_id(cls, model_id: str) -> bool:
        return model_id.startswith(cls.family()) and "embed" not in model_id

    @classmethod
    def family(cls) -> str:
        return "stability.stable"

    def get_body(
        self,
        prompts: List[Tuple],
        width: int,
        height: int,
        seed: int,
        **kwargs,
    ) -> str:
        # if (width, height) not in resolutions:
        #     raise BedrockExtraArgsError(f"Invalid resolution: {(width, height)}")

        samples = kwargs.get("samples", 1)
        if samples != 1:
            raise BedrockExtraArgsError("SDXL does not support multiple samples")
        body = {
            "text_prompts": [
                {"text": p[0], "weight": p[1] if len(p) > 1 else 1} for p in prompts
            ],
            "seed": seed,
        }

        body["samples"] = samples
        body["sampler_name"] = kwargs.get("sampler", None)
        body["steps"] = kwargs.get("steps", 50)
        body["width"] = width
        body["height"] = height
        if "clip_guidance_preset" in kwargs:
            body["clip_guidance_preset"] = kwargs["clip_guidance_preset"]
        if "style_preset" in kwargs:
            body["style_preset"] = kwargs["style_preset"]
        if "cfg_scale" in kwargs:
            body["cfg_scale"] = kwargs["cfg_scale"]

        return json.dumps(body)

    def generate(
        self,
        prompts: List[Tuple],
        height: int = 512,
        width: int = 512,
        seed: int = 1,
        *,
        samples: int = 1,
        sampler: Optional[str] = None,
        steps: int = 50,
        style_preset: SDStylePresets = None,
        clip_guidance_preset: str = "NONE",
        cfg_scale: int = 7,
    ) -> List[Image.Image]:
        if style_preset:
            return super()._generate(
                prompts,
                height,
                width,
                seed,
                samples=samples,
                sampler=sampler,
                steps=steps,
                cfg_scale=cfg_scale,
                clip_guidance_preset=clip_guidance_preset,
                style_preset=style_preset.value,
            )
        else:
            return super()._generate(
                prompts,
                height,
                width,
                seed,
                samples=samples,
                sampler=sampler,
                steps=steps,
                cfg_scale=cfg_scale,
                clip_guidance_preset=clip_guidance_preset,
            )

    def get_images(self, resp: Dict[str, Any]) -> List[Image.Image]:
        body_json = json.loads(resp["body"].read())

        imgs = [
            Image.open(BytesIO(b64decode(v["base64"]))) for v in body_json["artifacts"]
        ]

        return imgs
