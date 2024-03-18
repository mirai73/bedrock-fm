from typing import Any, Dict, List, Tuple, Optional
from enum import Enum
from .bedrock import BedrockFoundationModel, Model
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
    def __init__(self):
        super().__init__()
        self._model_id = Model.AMAZON_TITAN_IMAGE_GENERATOR_V1.value

    def get_body(
        self,
        prompts: List[Tuple],
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        negative_prompt: Optional[str] = None,
        cfg_scale: int = 7.0,
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
        return json.dumps(body)

    def generate(
        self,
        prompts: List[Tuple],
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        *,
        negative_prompt: Optional[str] = None,
        cfg_scale: int = 7.0,
        number_of_images: int = 1,
    ) -> List[Image.Image]:
        return super()._generate(
            prompts,
            height=height,
            width=width,
            seed=seed,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            number_of_images=number_of_images,
        )


@define
class TitanImageVariation(TitanImageBase):
    def __init__(self):
        super().__init__()
        self._model_id = Model.AMAZON_TITAN_IMAGE_GENERATOR_V1.value

    def get_body(
        self,
        prompts: List[Tuple],
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        *,
        images: List[Image.Image],
        negative_prompt: Optional[str] = None,
        cfg_scale: int = 7.0,
        number_of_images: int = 1,
    ) -> str:
        if len(images) == 0:
            raise ValueError(
                "You need to provide at least one image for the parameter images="
            )
        b64_im = []
        for im in images:
            buffer = BytesIO()
            im.save(buffer, format="png")
            buf = buffer.getvalue()
            b64_im.append(str(b64encode(buf), "ascii"))

        body = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "text": prompts[0][0],
                "images": b64_im,
            },
            "imageGenerationConfig": {"seed": seed},
        }
        if negative_prompt != None:
            body["imageVariationParams"]["negativeText"] = negative_prompt

        body["imageGenerationConfig"]["cfgScale"] = cfg_scale
        body["imageGenerationConfig"]["numberOfImages"] = number_of_images
        body["imageGenerationConfig"]["height"] = height
        body["imageGenerationConfig"]["width"] = width

        return json.dumps(body)

    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        *,
        images: List[Image.Image],
        negative_prompt: Optional[str] = None,
        cfg_scale: int = 7,
        number_of_images: int = 1,
    ) -> List[Image.Image]:
        return super()._generate(
            [(prompt,)],
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            seed=seed,
            images=images,
            cfg_scale=cfg_scale,
            number_of_images=number_of_images,
        )


@define
class TitanImageInPainting(TitanImageBase):
    def __init__(self):
        super().__init__()
        self._model_id = Model.AMAZON_TITAN_IMAGE_GENERATOR_V1.value

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
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        *,
        image: Image.Image,
        mask_image: Optional[Image.Image] = None,
        mask_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        cfg_scale: int = 7,
        number_of_images: int = 1,
    ) -> List[Image.Image]:
        return super()._generate(
            [(prompt,)],
            mask_image=mask_image,
            mask_prompt=mask_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            seed=seed,
            image=image,
            cfg_scale=cfg_scale,
            number_of_images=number_of_images,
        )


class OutpaintingMode(Enum):
    DEFAULT = "DEFAULT"
    PRECISE = "PRECISE"


@define
class TitanImageOutPainting(TitanImageBase):
    def __init__(self):
        super().__init__()
        self._model_id = Model.AMAZON_TITAN_IMAGE_GENERATOR_V1.value

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
        outpainting_mode: Optional[OutpaintingMode] = OutpaintingMode.DEFAULT,
        cfg_scale: int = 7.0,
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
            "taskType": "OUTPAINTING",
            "outPaintingParams": {
                "text": prompts[0][0],
                "image": str(b64encode(buf), "ascii"),
            },
            "imageGenerationConfig": {"seed": seed},
        }
        if negative_prompt != None:
            body["outPaintingParams"]["negativeText"] = negative_prompt

        body["imageGenerationConfig"]["cfgScale"] = cfg_scale
        body["imageGenerationConfig"]["numberOfImages"] = number_of_images
        body["imageGenerationConfig"]["height"] = height
        body["imageGenerationConfig"]["width"] = width
        body["outPaintingParams"]["outPaintingMode"] = outpainting_mode.value

        if mask_prompt != None:
            body["outPaintingParams"]["maskPrompt"] = mask_prompt

        if mask_image != None:
            buffer = BytesIO()
            image.save(buffer, format="png")
            buf = buffer.getvalue()
            body["outPaintingParams"]["maskImage"] = str(b64encode(buf), "ascii")

        return json.dumps(body)

    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        *,
        image: Image.Image,
        cfg_scale: int = 7,
        number_of_images: int = 1,
        mask_image: Optional[Image.Image] = None,
        mask_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> List[Image.Image]:
        return super()._generate(
            [(prompt,)],
            mask_image=mask_image,
            mask_prompt=mask_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            seed=seed,
            image=image,
            cfg_scale=cfg_scale,
            number_of_images=number_of_images,
        )
