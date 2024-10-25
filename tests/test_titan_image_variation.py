from bedrock_fm import TitanImageVariation, Model
import json
from PIL import Image

fm = TitanImageVariation.from_id(Model.AMAZON_TITAN_IMAGE_GENERATOR_V1)


def test_args():
    b = fm.get_body(
        [("fruits", 1)],
        512,
        512,
        0,
        images=[Image.new(mode="RGB", size=(512, 512))],
    )
    b_obj = json.loads(b)
    assert list(b_obj.keys()) == [
        "taskType",
        "imageVariationParams",
        "imageGenerationConfig",
    ]
    assert b_obj["taskType"] == "IMAGE_VARIATION"
    assert b_obj["imageGenerationConfig"] == {
        "cfgScale": 7.0,
        "height": 512,
        "width": 512,
        "numberOfImages": 1,
        "seed": 0,
    }
    assert b_obj["imageVariationParams"]["text"] == "fruits"
    assert "images" in b_obj["imageVariationParams"]


im = Image.open("tests/test_image.png")


def test_gen():
    r = fm.generate(
        "windy, dark",
        512,
        512,
        0,
        images=[im],
        negative_prompt="dogs",
        number_of_images=1,
    )
    assert type(r) == list
    assert len(r) == 1
