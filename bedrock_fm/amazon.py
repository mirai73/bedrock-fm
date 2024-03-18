from typing import Any, Dict, List, Tuple, Optional
from .bedrock import BedrockFoundationModel
from .bedrock_image import BedrockImageModel
from .exceptions import BedrockExtraArgsError
from PIL import Image
import json
from attrs import define
from io import BytesIO
from base64 import b64decode, b64encode


@define
class Titan(BedrockFoundationModel):
    """Amazon Titan Foundation Models are pre-trained on large datasets, making them powerful, general-purpose models.
    Use them as is, or customize them by fine tuning the models with your own data for a particular task without annotating large volumes of data.

    Titan Text models are generative LLMs for tasks such as summarization, text generation (for example, creating a blog post), classification,
    open-ended Q&A,
    and information extraction. They are also trained on many different programming languages as well as rich text format like tables,
    JSON and csv’s among others.

    To continue supporting best practices in the responsible use of AI, Titan Foundation Models are built to detect and
    remove harmful content in the data, reject inappropriate content in the user input, and filter the models’ outputs that contain
    inappropriate content (such as hate speech, profanity, and violence).
    Titan models do not support any `extra_args`.
    """

    @classmethod
    def _validate_model_id(cls, model_id: str) -> bool:
        return model_id.startswith(cls.family()) and "embed" not in model_id

    @classmethod
    def family(cls) -> str:
        return "amazon.titan"

    def validate_extra_args(self, extra_args: Dict[str, Any]) -> None:
        if len(extra_args.keys()) > 0:
            raise BedrockExtraArgsError("This model does not support any extra_args")

    def get_body(
        self,
        prompt: str,
        top_p: float,
        temperature: float,
        max_token_count: int,
        stop_sequences: [str],
        extra_args: Dict[str, Any],
        stream: bool,
    ) -> str:
        return json.dumps(
            {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_token_count,
                    "stopSequences": stop_sequences,
                    "temperature": temperature,
                    "topP": top_p,
                },
            }
        )

    def process_response_body(self, out) -> List[str]:
        return [self.get_text(r) for r in out["results"]]

    def get_text(self, body: Dict[str, Any]) -> str:
        return body["outputText"]


@define
class TitanImageBase(BedrockImageModel):
    @classmethod
    def _validate_model_id(cls, model_id: str) -> bool:
        return model_id.startswith(cls.family()) and "embed" not in model_id

    @classmethod
    def family(cls) -> str:
        return "amazon.titan-image"

    def validate_extra_args(self, extra_args: Dict[str, Any]) -> bool:
        unsupp_args = []
        supported_args = [
            "numberOfImages",
            "height",
            "width",
            "cfgScale",
            "steps",
        ]
        for k in extra_args.keys():
            if k not in supported_args:
                unsupp_args.append(k)

        if len(unsupp_args) > 0:
            raise BedrockExtraArgsError(
                f"Arguments [{','.join(unsupp_args)}] are not supported by this model.\nOnly [{','.join(supported_args)}] are supported"
            )
        return True

    def get_images(self, resp: Dict[str, Any]) -> List[Image.Image]:
        body_json = json.loads(resp["body"].read())

        imgs = [Image.open(BytesIO(b64decode(v))) for v in body_json["images"]]

        return imgs


@define
class TitanImageGeneration(TitanImageBase):
    def get_body(
        self,
        prompts: List[Tuple],
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        negative_prompt: Optional[str] = None,
        cfg_scale: int = 7.0,
        quality: str = "standard",
        number_of_images: int = 1,
    ) -> str:
        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompts[0][0],
            },
            "imageGenerationConfig": {"seed": seed},
        }
        if negative_prompt != None:
            body["textToImageParams"]["negativeText"] = negative_prompt
        body["imageGenerationConfig"]["cfgScale"] = cfg_scale
        body["imageGenerationConfig"]["numberOfImages"] = number_of_images
        body["imageGenerationConfig"]["height"] = height
        body["imageGenerationConfig"]["width"] = width
        body["imageGenerationConfig"]["quality"] = quality
        return json.dumps(body)

    def generate(
        self,
        prompts: List[Tuple],
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        negative_prompt: Optional[str] = None,
        cfg_scale: int = 7.0,
        quality: str = "standard",
        number_of_images: int = 1,
    ) -> List[Image.Image]:
        return super().generate(
            prompts,
            height=height,
            width=width,
            seed=seed,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            quality=quality,
            number_of_images=number_of_images,
        )


@define
class TitanImageVariation(TitanImageBase):
    def get_body(
        self,
        prompts: List[Tuple],
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        negative_prompt: Optional[str] = None,
        image: Optional[Image.Image] = None,
        cfg_scale: int = 7.0,
        quality: str = "standard",
        number_of_images: int = 1,
    ) -> str:
        buffer = BytesIO()
        image.save(buffer, format="png")
        buf = buffer.getvalue()
        body = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "text": prompts[0][0],
                "images": [str(b64encode(buf), "ascii")],
            },
            "imageGenerationConfig": {"seed": seed},
        }
        if negative_prompt != None:
            body["imageVariationParams"]["negativeText"] = negative_prompt

        body["imageGenerationConfig"]["cfgScale"] = cfg_scale
        body["imageGenerationConfig"]["numberOfImages"] = number_of_images
        body["imageGenerationConfig"]["height"] = height
        body["imageGenerationConfig"]["width"] = width
        body["imageGenerationConfig"]["quality"] = quality

        return json.dumps(body)

    def generate(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        cfg_scale: int = 7,
        quality: str = "standard",
        number_of_images: int = 1,
    ) -> List[Image.Image]:
        return super().generate(
            [(prompt,)],
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            seed=seed,
            image=image,
            cfg_scale=cfg_scale,
            quality=quality,
            number_of_images=number_of_images,
        )


@define
class TitanImageInPainting(TitanImageBase):
    def get_body(
        self,
        prompts: List[Tuple],
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        mask_image: Optional[Image.Image] = None,
        mask_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        image: Optional[Image.Image] = None,
        cfg_scale: int = 7.0,
        quality: str = "standard",
        number_of_images: int = 1,
    ) -> str:
        if mask_prompt == None and mask_image == None:
            raise BedrockExtraArgsError("You must provide a mask prompt or mask image")
        if mask_prompt != None and mask_image != None:
            raise BedrockExtraArgsError(
                "You must provide either a mask prompt or a mask image"
            )
        buffer = BytesIO()
        image.save(buffer, format="png")
        buf = buffer.getvalue()
        body = {
            "taskType": "INPAINTING",
            "inPaintingParams": {
                "text": prompts[0][0],
                "image": str(b64encode(buf), "ascii"),
            },
            "imageGenerationConfig": {"seed": seed},
        }
        if negative_prompt != None:
            body["inPaintingParams"]["negativeText"] = negative_prompt

        body["imageGenerationConfig"]["cfgScale"] = cfg_scale
        body["imageGenerationConfig"]["numberOfImages"] = number_of_images
        body["imageGenerationConfig"]["height"] = height
        body["imageGenerationConfig"]["width"] = width
        body["imageGenerationConfig"]["quality"] = quality

        if mask_prompt != None:
            body["inPaintingParams"]["maskPrompt"] = mask_prompt

        if mask_image != None:
            buffer = BytesIO()
            image.save(buffer, format="png")
            buf = buffer.getvalue()
            body["inPaintingParams"]["maskImage"] = str(b64encode(buf), "ascii")

        return json.dumps(body)

    def generate(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Optional[Image.Image] = None,
        mask_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        cfg_scale: int = 7,
        quality: str = "standard",
        number_of_images: int = 1,
    ) -> List[Image.Image]:
        return super().generate(
            [(prompt,)],
            mask_image=mask_image,
            mask_prompt=mask_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            seed=seed,
            image=image,
            cfg_scale=cfg_scale,
            quality=quality,
            number_of_images=number_of_images,
        )


# {
#   "modelId": "amazon.titan-image-generator-v1:0",
#   "contentType": "application/json",
#   "accept": "application/json",
#   "body": "{
#     "inPaintingParams": {
#       "text": "change flowers to orange"
#       "image": [
#         "iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf..."
#       ]
#     },
#     "taskType": "IMAGE_VARIATION",
#     "imageGenerationConfig": {
#       "cfgScale": 8,
#       "seed": 0,
#       "quality": "standard",
#       "width": 1024,
#       "height": 1024,
#       "numberOfImages": 3
#     }
#   }"
# }
