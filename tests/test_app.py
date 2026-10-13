import pytest
import numpy as np
from PIL import Image
from utils.utils import load_image, image_preprocess, resize_image_proportionally


@pytest.fixture
def test_image(tmp_path):
    image_path = tmp_path / "test_image.png"
    image = Image.new("RGB", (100, 100), color=(73, 109, 137))
    image.save(image_path)
    return image_path


def test_load_image(test_image):
    img = load_image(test_image, size=64)
    assert img.size == (64, 64)

    img = load_image(test_image, scale=2)
    assert img.size == (50, 50)


def test_load_image_accepts_pil_image():
    img = load_image(Image.new("RGBA", (600, 300)))
    assert img.mode == "RGB"
    assert img.size == (512, 256)


def test_load_image_applies_exif_orientation(tmp_path):
    # Orientation 6 means "rotate 90° clockwise to display", as phones store portrait shots
    image_path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (100, 50)).save(image_path, exif=exif)

    img = load_image(image_path)
    assert img.size == (50, 100)


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
