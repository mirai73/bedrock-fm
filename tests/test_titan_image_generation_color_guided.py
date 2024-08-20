from bedrock_fm import TitanImageColorGuidedContent, Model
import json
from PIL import Image

fm = TitanImageColorGuidedContent.from_id(Model.AMAZON_TITAN_IMAGE_GENERATOR_V2_0)


def test_args():
    b = fm.get_body([("hello", 1)], 512, 512, 0, colors=["#ff8080", "#ffb280"])
    assert json.loads(b) == {
        "taskType": "COLOR_GUIDED_GENERATION",
        "colorGuidedGenerationParams": {
            "colors": ["#ff8080", "#ffb280"],
            "text": "hello",
        },
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
        "hello",
        512,
        512,
        0,
        negative_prompt="dogs",
        number_of_images=2,
        reference_image=Image.new("RGB", (512, 512)),
        colors=["#ff8080", "#ffb280"],
    )
    assert type(r) == list
    assert len(r) == 1
