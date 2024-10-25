from bedrock_fm import TitanImageColorGuidedContent, Model
import json
from PIL import Image

fm = TitanImageColorGuidedContent.from_id(Model.AMAZON_TITAN_IMAGE_GENERATOR_V2_0)


def test_args():
    b = fm.get_body([("fruits", 1)], 512, 512, 0, colors=["#ff8080", "#ffb280"])
    assert json.loads(b) == {
        "taskType": "COLOR_GUIDED_GENERATION",
        "colorGuidedGenerationParams": {
            "colors": ["#ff8080", "#ffb280"],
            "text": "fruits",
        },
        "imageGenerationConfig": {
            "cfgScale": 7.0,
            "height": 512,
            "numberOfImages": 1,
            "seed": 0,
            "width": 512,
        },
    }


im = Image.open("tests/test_image.png")


def test_gen():
    r = fm.generate(
        "tropical animals in the forest",
        512,
        512,
        0,
        number_of_images=1,
        reference_image=im,
        colors=["#ff8080", "#ffb280"],
    )
    assert type(r) == list
    assert len(r) == 1
