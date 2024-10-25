from bedrock_fm import TitanImageGeneration, Model
import json
from PIL import Image

fm = TitanImageGeneration.from_id(Model.AMAZON_TITAN_IMAGE_GENERATOR_V1)


def test_args():
    b = fm.get_body([("hello", 1)], 512, 512, 0)
    assert json.loads(b) == {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": "hello"},
        "imageGenerationConfig": {
            "cfgScale": 7.0,
            "height": 512,
            "numberOfImages": 1,
            "seed": 0,
            "width": 512,
        },
    }


def test_gen():
    r = fm.generate(
        [("animals", 1)], 512, 512, 0, negative_prompt="dogs", number_of_images=2
    )
    assert type(r) == list
    assert len(r) == 2
