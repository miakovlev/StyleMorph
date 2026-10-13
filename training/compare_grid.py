"""Grid of the test images stylized by several models, one column per model.

    python -m training.compare_grid --out runs/starry_compare.jpg \\
        old=models/epoch_500_1_100000.0_style2.onnx new=runs/starry/best.model

Models are label=path pairs; .onnx files run through ONNX Runtime exactly as in the app,
.model checkpoints through PyTorch with the same padding. Images are downscaled to
512 px on the long side before stylization, as in the app.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.utils import image_preprocess, load_image, stylize

SPLITS = Path(__file__).resolve().parent / "splits" / "coco_val2017"


def stylize_checkpoint(image, model):
    import torch

    x = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float().unsqueeze(0)
    height, width = x.shape[2:]
    x = torch.nn.functional.pad(x, (0, -width % 4, 0, -height % 4), mode="reflect")
    with torch.no_grad():
        output = model(x)[:, :, :height, :width]
    return image_preprocess(output.numpy())


def make_stylizer(path):
    if path.endswith(".onnx"):
        return lambda image: image_preprocess(stylize(image, path))
    from training.export_onnx import load_transformer
    model = load_transformer(path)
    return lambda image: stylize_checkpoint(image, model)


def build_grid(image_paths, models, cell_width):
    stylizers = {label: make_stylizer(path) for label, path in models.items()}
    labels = ["original", *stylizers]
    font_size = max(14, cell_width // 16)
    font = ImageFont.load_default(size=font_size)
    header = 2 * font_size

    rows = []
    for image_path in image_paths:
        image = load_image(image_path)
        cells = [image] + [stylizer(image) for stylizer in stylizers.values()]
        height = round(cell_width * image.height / image.width)
        rows.append([cell.resize((cell_width, height), Image.LANCZOS) for cell in cells])

    grid = Image.new("RGB", (cell_width * len(labels), header + sum(row[0].height for row in rows)), "white")
    draw = ImageDraw.Draw(grid)
    for column, label in enumerate(labels):
        draw.text((column * cell_width + cell_width // 2, header // 2), label, fill="black", font=font, anchor="mm")
    top = header
    for row in rows:
        for column, cell in enumerate(row):
            grid.paste(cell, (column * cell_width, top))
        top += row[0].height
    return grid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", metavar="label=path", help=".onnx or .model file to compare")
    parser.add_argument("--out", required=True, help="grid image to write")
    parser.add_argument("--images-dir", default="data/coco/val2017")
    parser.add_argument("--image-list", default=str(SPLITS / "test.txt"))
    parser.add_argument("--cell-width", type=int, default=384)
    args = parser.parse_args(argv)

    models = dict(model.split("=", 1) for model in args.models)
    image_paths = [Path(args.images_dir) / name for name in Path(args.image_list).read_text().split()]
    grid = build_grid(image_paths, models, args.cell_width)
    grid.save(args.out, quality=90)
    print(f"{args.out}: {len(image_paths)} images × {len(models)} models, {grid.width}×{grid.height} px")


if __name__ == "__main__":
    main()
