from bedrock_fm import TitanImageOutPainting, Model
import json
from PIL import Image

fm = TitanImageOutPainting.from_id(Model.AMAZON_TITAN_IMAGE_GENERATOR_V1)


def test_args():
    b = fm.get_body(
        [("hello", 1)],
        512,
        512,
        0,
        image=Image.new(mode="RGB", size=(512, 512)),
        mask_prompt="pears",
    )
    b_obj = json.loads(b)
    assert list(b_obj.keys()) == [
        "taskType",
        "outPaintingParams",
        "imageGenerationConfig",
    ]
    assert b_obj["taskType"] == "OUTPAINTING"
    assert b_obj["imageGenerationConfig"] == {
        "cfgScale": 7.0,
        "height": 512,
        "width": 512,
        "numberOfImages": 1,
        "seed": 0,
    }
    assert b_obj["outPaintingParams"]["text"] == "hello"
    assert "image" in b_obj["outPaintingParams"]
    assert b_obj["outPaintingParams"]["maskPrompt"] == "pears"


def test_gen():
    r = fm.generate(
        "cats",
        512,
        512,
        0,
        image=Image.new(mode="RGB", size=(512, 512)),
        negative_prompt="dogs",
        mask_prompt="black",
        number_of_images=1,
    )
    assert type(r) == list
    assert len(r) == 1
