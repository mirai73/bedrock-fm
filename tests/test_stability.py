from bedrock_fm import SDXL, SDStylePresets, Model
from bedrock_fm.exceptions import BedrockExtraArgsError
import json
from PIL import Image

fm = SDXL.from_id(Model.STABILITY_STABLE_DIFFUSION_XL)


def test_args():
    b = fm.get_body([("dog", 1)], 512, 512, 0)
    assert json.loads(b) == {
        "text_prompts": [{"text": "dog", "weight": 1}],
        "seed": 0,
        "samples": 1,
        "sampler_name": None,
        "steps": 50,
        "width": 512,
        "height": 512,
    }


def test_gen():
    r = fm.generate([("dog", 1)], 512, 512, 0)
    assert type(r) is list
    i = r[0]
    assert i.width == 512
    assert i.height == 512


def test_gen_attr():
    r = fm.generate([("dog", 1)], 512, 512, 0, style_preset=SDStylePresets.ANALOG_FILM)
    assert type(r) is list
    i = r[0]
    assert i.width == 512
    assert i.height == 512
