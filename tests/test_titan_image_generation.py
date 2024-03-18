from bedrock_fm import TitanImageGeneration
import json

fm = TitanImageGeneration()


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
        [("hello", 1)], 512, 512, 0, negative_prompt="dogs", number_of_images=2
    )
    assert type(r) == list
    assert len(r) == 2
