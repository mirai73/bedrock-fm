from bedrock_fm import TitanImageBackgroundRemoval, Model
import json
from PIL import Image

fm = TitanImageBackgroundRemoval.from_id(Model.AMAZON_TITAN_IMAGE_GENERATOR_V2_0)


def test_args():
    b = fm.get_body(
        [("hello", 1)],
        512,
        512,
        0,
        image=Image.new("RGB", (1, 1)),
    )
    assert json.loads(b) == {
        "taskType": "BACKGROUND_REMOVAL",
        "backgroundRemovalParams": {
            "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
        },
        "imageGenerationConfig": {
            "cfgScale": 7.0,
            "height": 512,
            "seed": 0,
            "width": 512,
        },
    }


im = Image.open("tests/test_image.png")


def test_gen():
    r = fm.generate(
        "a cat bathing in the sun",
        512,
        512,
        0,
        image=im,
    )
    assert type(r) == list
    assert len(r) == 1
