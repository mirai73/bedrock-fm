from bedrock_fm import TitanImageConditionedGeneration, Model, ControlMode
import json
from PIL import Image

fm = TitanImageConditionedGeneration.from_id(Model.AMAZON_TITAN_IMAGE_GENERATOR_V2_0)


def test_args():
    b = fm.get_body([("fruits", 1)], 512, 512, 0)
    assert json.loads(b) == {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": "fruits"},
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
        "animals",
        512,
        512,
        0,
        negative_prompt="dogs",
        number_of_images=2,
        condition_image=Image.new("RGB", (512, 512)),
        control_mode=ControlMode.CANNY_EDGE,
        control_strength=0.4,
    )
    assert type(r) == list
    assert len(r) == 2
