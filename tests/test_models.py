import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from utils.utils import stylize

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MODELS = sorted((ROOT / "models").glob("*.onnx"))
RETRAINED_MODELS = sorted((ROOT / "models" / "v2").glob("*.onnx"))
MODEL_PATHS = ORIGINAL_MODELS + RETRAINED_MODELS


def test_models_found():
    assert len(ORIGINAL_MODELS) == 7
    assert len(RETRAINED_MODELS) == 7


def test_app_model_paths_exist():
    paths = re.findall(r'"(models/[^"]+\.onnx)"', (ROOT / "app.py").read_text())
    assert len(paths) == 14
    assert len(set(paths)) == 14
    for path in paths:
        assert (ROOT / path).exists(), path


@pytest.mark.parametrize("model_path", MODEL_PATHS, ids=lambda path: str(path.relative_to(ROOT / "models")))
@pytest.mark.parametrize("size", [(64, 48), (67, 45)])
def test_stylize_keeps_image_size(model_path, size):
    pixels = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    output = stylize(Image.fromarray(pixels), str(model_path))

    assert output.shape == (1, 3, size[1], size[0])
    assert np.isfinite(output).all()


def test_app_renders():
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert app.radio[0].value == "Retrained (2026)"

    app.radio[0].set_value("Original (2024)").run()
    assert not app.exception
