from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from utils.utils import stylize

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = sorted((ROOT / "models").glob("*.onnx"))


def test_models_found():
    assert len(MODEL_PATHS) == 7


@pytest.mark.parametrize("model_path", MODEL_PATHS, ids=lambda path: path.stem)
@pytest.mark.parametrize("size", [(64, 48), (67, 45)])
def test_stylize_keeps_image_size(model_path, size):
    pixels = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    output = stylize(Image.fromarray(pixels), str(model_path))

    assert output.shape == (1, 3, size[1], size[0])
    assert np.isfinite(output).all()


def test_app_renders():
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
