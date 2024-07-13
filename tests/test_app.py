import pytest
import numpy as np
from PIL import Image
import os
from utils.utils import load_image, image_preprocess, resize_image_proportionally


@pytest.fixture
def test_image():
    image_path = "output_image.png"
    image = Image.new("RGB", (100, 100), color=(73, 109, 137))
    image.save(image_path)
    yield image_path
    if os.path.exists(image_path):
        os.remove(image_path)


def test_load_image(test_image):
    img = load_image(test_image, size=64)
    assert img.size == (64, 64)

    img = load_image(test_image, scale=2)
    assert img.size == (50, 50)


def test_image_preprocess():
    tensor = np.random.rand(1, 3, 100, 100).astype(np.float32)
    image = image_preprocess(tensor)
    assert isinstance(image, Image.Image)
    assert image.size == (100, 100)


def test_resize_image_proportionally(test_image):
    image = Image.open(test_image)
    resized_image = resize_image_proportionally(image, 50)
    assert resized_image.size == (50, 50)

    resized_image = resize_image_proportionally(image, 200)
    assert resized_image.size == (200, 200)
