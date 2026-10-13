from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Training needs torch from requirements-train.txt; without it these tests are skipped
torch = pytest.importorskip("torch")

from training.compare_grid import build_grid, stylize_checkpoint  # noqa: E402
from training.export_onnx import export, load_transformer, max_difference  # noqa: E402
from training.make_splits import make_splits  # noqa: E402
from training.train import parse_args, train  # noqa: E402
from utils.utils import image_preprocess, stylize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = sorted((ROOT / "models").glob("*.model"))
STARRY_NIGHT = ROOT / "models" / "epoch_500_1_100000.0_style2"


@pytest.fixture
def image_folder(tmp_path):
    rng = np.random.default_rng(0)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(10):
        pixels = rng.integers(0, 256, (40 + i, 50, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images_dir / f"{i:02d}.jpg")
    return images_dir


@pytest.fixture
def splits(image_folder, tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "test.txt").write_text("00.jpg\n01.jpg\n")
    make_splits(image_folder, splits_dir, val_size=3)
    return splits_dir


def test_old_checkpoints_load():
    assert len(CHECKPOINTS) == 7
    for checkpoint in CHECKPOINTS:
        load_transformer(checkpoint)


def test_make_splits_keeps_test_images_out(splits):
    train_names = set((splits / "train.txt").read_text().split())
    val_names = set((splits / "val.txt").read_text().split())

    assert len(train_names) == 5
    assert len(val_names) == 3
    assert not train_names & val_names
    assert not {"00.jpg", "01.jpg"} & (train_names | val_names)


def test_train_smoke(image_folder, splits, tmp_path):
    out = tmp_path / "run"
    args = parse_args([
        "--style", str(ROOT / "images" / "style_images" / "style2.jpg"), "--out", str(out),
        "--style-size", "64", "--images", "12", "--batch-size", "2", "--image-size", "32", "--val-every", "3",
        "--images-dir", str(image_folder), "--train-list", str(splits / "train.txt"),
        "--val-list", str(splits / "val.txt"), "--sample-list", str(splits / "test.txt"), "--sample-size", "32",
        "--device", "cpu", "--workers", "0", "--no-pretrained-vgg",
    ])
    rows = train(args)

    # 5 training images give 2 batches per pass, so 6 steps take three passes over the data
    assert [row["step"] for row in rows] == [3, 6]
    assert all(np.isfinite(row["val_total"]) for row in rows)
    for name in ["config.json", "losses.csv", "losses.png", "best.model", "last.model", "samples/step_000006.jpg"]:
        assert (out / name).exists(), name
    load_transformer(out / "best.model")


def test_export_matches_pytorch(tmp_path):
    model = load_transformer(f"{STARRY_NIGHT}.model")
    path = str(tmp_path / "model.onnx")
    export(model, path)

    assert max_difference(model, path, sizes=[(64, 64), (48, 68)]) < 0.01
    # The app pads odd sizes to a multiple of 4 and crops back
    assert stylize(Image.new("RGB", (67, 45)), path).shape == (1, 3, 45, 67)


def test_checkpoint_matches_app_model():
    # The committed .model checkpoints are the sources of the ONNX models the app runs
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (45, 67, 3), dtype=np.uint8))
    from_checkpoint = stylize_checkpoint(image, load_transformer(f"{STARRY_NIGHT}.model"))
    from_onnx = image_preprocess(stylize(image, f"{STARRY_NIGHT}.onnx"))

    difference = np.abs(np.asarray(from_checkpoint, dtype=int) - np.asarray(from_onnx, dtype=int))
    assert difference.max() <= 1


def test_compare_grid(image_folder):
    models = {"onnx": f"{STARRY_NIGHT}.onnx", "checkpoint": f"{STARRY_NIGHT}.model"}
    grid = build_grid([image_folder / "00.jpg", image_folder / "09.jpg"], models, cell_width=64)

    # Original plus two models; rows keep each image's aspect ratio under a label header
    assert grid.width == 3 * 64
    assert grid.height == 2 * 14 + round(64 * 40 / 50) + round(64 * 49 / 50)
