from bedrock_fm import TitanImageInPainting, Model
import json
from PIL import Image

fm = TitanImageInPainting.from_id(Model.AMAZON_TITAN_IMAGE_GENERATOR_V1)


def test_args():
    b = fm.get_body(
        [("hello", 1)],
        512,
        512,
        0,
        mask_prompt="pears",
        image=Image.new("RGB", (512, 512)),
    )
    b_obj = json.loads(b)
    assert list(b_obj.keys()) == [
        "taskType",
        "inPaintingParams",
        "imageGenerationConfig",
    ]
    assert b_obj["taskType"] == "INPAINTING"
    assert b_obj["imageGenerationConfig"] == {
        "cfgScale": 7.0,
        "height": 512,
        "width": 512,
        "numberOfImages": 1,
        "seed": 0,
    }
    assert b_obj["inPaintingParams"]["text"] == "hello"
    assert "image" in b_obj["inPaintingParams"]
    assert b_obj["inPaintingParams"]["maskPrompt"] == "pears"


def test_gen():
    r = fm.generate(
        "cats",
        512,
        512,
        0,
        image=Image.new("RGB", (512, 512)),
        negative_prompt="dogs",
        number_of_images=1,
        mask_prompt="black",
    )
    assert type(r) == list
    assert len(r) == 1
