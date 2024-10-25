"""Higher level SDK to interact with [Amazon Bedrock]().

The `bedrock_fm` library exposes a separate class for each of the Bedrock models, with an simpler `generate()` API which is common across all models. The same method can be used to get a stream instead of a full completion by passing the `stream=True` as parameter. To obtain a detailed response including the original prompt, the body passed to the `invoke_*` method and timing information you can pass the parameter `details=True`. The API is fully typed, including the different return types based on the `stream` and `details` parameters values. 

The output generation can be tuned with the optional `temperature`, `top_p` and `stop_words` parameters which can be passed at the instance creation time (in the class constructor) and overridden at generation time in the `generate` method.
"""

from .amazon import (
    Titan,
    TitanImageGeneration,
    TitanImageVariation,
    TitanImageInPainting,
    TitanImageOutPainting,
    TitanImageConditionedGeneration,
    TitanImageColorGuidedContent,
    TitanImageBackgroundRemoval,
    ControlMode,
    OutpaintingMode,
)
from .anthropic import Claude, Claude3
from .ai21 import Jurassic, Jamba
from .cohere import Command, CommandR
from .meta import Llama2Chat, Llama3Instruct
from .cohere_embeddings import Embed
from .mistral import Mistral, Mixtral, MistralLarge
from .titan_embeddings import TitanEmbeddings
from .stability import SDXL, SDStylePresets
from .bedrock import (
    StreamDetails,
    CompletionDetails,
    BedrockFoundationModel,
    BedrockEmbeddingsModel,
    EmbeddingType,
    InstanceProfile,
)
from .bedrock_image import BedrockImageModel
from attrs import field
from .exceptions import BedrockInvalidModelError
from .bedrock import Human, Assistant, System
from .model import Model

__all__ = [
    "Titan",
    "TitanImageGeneration",
    "TitanImageVariation",
    "TitanImageInPainting",
    "TitanImageOutPainting",
    "TitanImageConditionedGeneration",
    "TitanImageColorGuidedContent",
    "TitanImageBackgroundRemoval",
    "ControlMode",
    "OutpaintingMode",
    "Claude",
    "Claude3",
    "Jurassic",
    "Jamba",
    "StreamDetails",
    "CompletionDetails",
    "BedrockFoundationModel",
    "BedrockEmbeddingsModel",
    "BedrockImageModel",
    "Command",
    "CommandR",
    "Embed",
    "Mistral",
    "Mixtral",
    "MistralLarge",
    "Llama2Chat",
    "Llama3Instruct",
    "SDXL",
    "SDStylePresets",
    "from_model_id",
    "Model",
    "Human",
    "Assistant",
    "System",
    "EmbeddingType",
    "InstanceProfile",
]


__family_map = {
    Titan.family(): Titan,
    Command.family(): Command,
    Jurassic.family(): Jurassic,
    Claude.family(): Claude,
    TitanEmbeddings.family(): TitanEmbeddings,
    Embed.family(): Embed,
    Llama2Chat.family(): Llama2Chat,
    Embed.family(): Embed,
    SDXL.family(): SDXL,
    Claude3.family(): Claude3,
    Jamba.family(): Jamba,
}


def from_model_id(
    model_id: str | Model, **kwargs
) -> BedrockFoundationModel | BedrockEmbeddingsModel | BedrockImageModel:
    """Instantiates a Bedrock Foundation Model or Embedding Model based on the `model_id`.

    Usage example:

    ```py
    from bedrock_fm import from_model_id

    fm = from_model_id("amazon.titan-embed-g1-text-02")
    # fm is of type `bedrock_fm.Titan`
    ```

    Args:
        model_id (str): the Amazon Bedrock [modelId]()

    Raises:
        BedrockInvalidModelError:

    Returns:
        BedrockFoundationModel | BedrockEmbeddingsModel: A typed instance of the Foundation Model
    """

    if type(model_id) == Model:
        model_id = model_id.value

    if "-" not in model_id:
        raise BedrockInvalidModelError(f"{model_id} is not a supported model")
    family = model_id.split("-")[0]
    if family == "amazon.titan" and "embed" in model_id:
        family += "-embed"  # Amazon Titan model naming workaround
    if family == "anthropic.claude" and "-3" in model_id:
        family += "-3"
    if family == "cohere.command" and "-r" in model_id:
        family += "-r"
    if family == "mistral.mistral" and "-large" in model_id:
        family += "-large"

    if family in __family_map:
        return __family_map[family].from_id(model_id, **kwargs)
    raise BedrockInvalidModelError(f"{model_id} is not a supported model")
