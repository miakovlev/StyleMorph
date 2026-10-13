"""Train a fast style transfer network (Johnson et al., 2016) for one style image.

Adapted from the PyTorch fast_neural_style example and StylusFusion.ipynb. Example:

    python -m training.train --style images/style_images/style2.jpg --out runs/starry

The run directory gets config.json, losses.csv, losses.png, best.model (lowest
validation loss), last.model and samples/ with the test images at every validation.
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import save_image

from utils.transformer_net import TransformerNet
from utils.utils import resize_image_proportionally

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = Path(__file__).resolve().parent / "splits" / "coco_val2017"


class ImageListDataset(Dataset):
    """Images named one per line in `list_file`, resized and center-cropped to a square."""

    def __init__(self, images_dir, list_file, image_size):
        self.images_dir = Path(images_dir)
        self.names = Path(list_file).read_text().split()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image = Image.open(self.images_dir / self.names[index]).convert("RGB")
        # Pixel values stay in 0-255, as the app feeds them to the ONNX model
        return self.transform(image).float()


class Vgg16(nn.Module):
    """Frozen VGG16 returning the relu1_2, relu2_2, relu3_3 and relu4_3 feature maps."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        features = models.vgg16(weights=weights).features
        self.slices = nn.ModuleList([features[:4], features[4:9], features[9:16], features[16:23]])
        self.requires_grad_(False)

    def forward(self, x):
        outputs = []
        for vgg_slice in self.slices:
            x = vgg_slice(x)
            outputs.append(x)
        return outputs


def normalize_batch(batch):
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, -1, 1, 1)
    return (batch / 255.0 - mean) / std


def gram_matrix(features):
    b, ch, h, w = features.size()
    features = features.view(b, ch, h * w)
    return features.bmm(features.transpose(1, 2)) / (ch * h * w)


class PerceptualLoss(nn.Module):
    """Weighted content loss (relu2_2) and style loss (Gram matrices of all four VGG layers)."""

    def __init__(self, vgg, style_image, content_weight, style_weight):
        super().__init__()
        self.vgg = vgg
        self.content_weight = content_weight
        self.style_weight = style_weight
        with torch.no_grad():
            self.style_grams = [gram_matrix(f) for f in vgg(normalize_batch(style_image))]

    def forward(self, content, stylized):
        with torch.no_grad():
            content_features = self.vgg(normalize_batch(content))
        stylized_features = self.vgg(normalize_batch(stylized))

        content_loss = F.mse_loss(stylized_features[1], content_features[1])
        style_loss = sum(F.mse_loss(gram_matrix(f), gram.expand(f.size(0), -1, -1))
                         for f, gram in zip(stylized_features, self.style_grams))
        return self.content_weight * content_loss, self.style_weight * style_loss


def load_style_image(path, style_size):
    image = Image.open(path).convert("RGB")
    if style_size:
        image = resize_image_proportionally(image, style_size)
    return transforms.functional.pil_to_tensor(image).float().unsqueeze(0)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(transformer, loader, loss_fn, device):
    transformer.eval()
    totals, count = torch.zeros(2), 0
    for x in loader:
        x = x.to(device)
        content_loss, style_loss = loss_fn(x, transformer(x))
        totals += torch.tensor([content_loss.item(), style_loss.item()]) * len(x)
        count += len(x)
    transformer.train()
    return (totals / count).tolist()


@torch.no_grad()
def save_samples(transformer, images, path):
    transformer.eval()
    stylized = transformer(images).clamp(0, 255)
    transformer.train()
    # Originals on the top row, stylized images below
    save_image(torch.cat([images, stylized]) / 255, path, nrow=len(images))


def plot_losses(rows, path):
    images = [row["images"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, ["content", "style", "total"]):
        ax.plot(images, [row[f"train_{name}"] for row in rows], label="train")
        ax.plot(images, [row[f"val_{name}"] for row in rows], label="val")
        ax.set_title(f"{name} loss")
        ax.set_xlabel("images seen")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def state_dict_on_cpu(model):
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_set = ImageListDataset(args.images_dir, args.train_list, args.image_size)
    val_set = ImageListDataset(args.images_dir, args.val_list, args.image_size)
    sample_set = ImageListDataset(args.images_dir, args.sample_list, args.sample_size)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed), **loader_options)
    val_loader = DataLoader(val_set, **loader_options)
    samples = torch.stack([sample_set[i] for i in range(min(len(sample_set), args.samples))]).to(device)

    transformer = TransformerNet().to(device)
    if args.init:
        transformer.load_state_dict(torch.load(args.init, map_location="cpu"))
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)
    style_image = load_style_image(args.style, args.style_size).to(device)
    loss_fn = PerceptualLoss(Vgg16(args.pretrained_vgg).to(device), style_image,
                             args.content_weight, args.style_weight)

    total_steps = args.images // args.batch_size
    print(f"device {device}, train {len(train_set)}, val {len(val_set)} images, "
          f"{total_steps} steps of {args.batch_size}, style image {tuple(style_image.shape[2:])}", flush=True)

    columns = ["step", "images", "seconds", "images_per_sec", "train_content", "train_style", "train_total",
               "val_content", "val_style", "val_total"]
    rows = []
    with open(out / "losses.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        step, best_val = 0, float("inf")
        interval_losses, interval_steps = torch.zeros(2, device=device), 0
        start = interval_start = time.time()
        while step < total_steps:
            for x in train_loader:
                x = x.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                content_loss, style_loss = loss_fn(x, transformer(x))
                (content_loss + style_loss).backward()
                optimizer.step()

                # Accumulate on the device: calling .item() every step would stall the GPU
                interval_losses += torch.stack([content_loss.detach(), style_loss.detach()])
                interval_steps += 1
                step += 1

                if step % args.val_every == 0 or step == total_steps:
                    train_content, train_style = (interval_losses / interval_steps).tolist()
                    val_content, val_style = evaluate(transformer, val_loader, loss_fn, device)
                    now = time.time()
                    row = {
                        "step": step,
                        "images": step * args.batch_size,
                        "seconds": round(now - start, 1),
                        "images_per_sec": round(interval_steps * args.batch_size / (now - interval_start), 1),
                        "train_content": train_content,
                        "train_style": train_style,
                        "train_total": train_content + train_style,
                        "val_content": val_content,
                        "val_style": val_style,
                        "val_total": val_content + val_style,
                    }
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                    plot_losses(rows, out / "losses.png")
                    save_samples(transformer, samples, out / "samples" / f"step_{step:06d}.jpg")

                    is_best = row["val_total"] < best_val
                    if is_best:
                        best_val = row["val_total"]
                        torch.save(state_dict_on_cpu(transformer), out / "best.model")
                    print(f"step {step}/{total_steps}  {row['images_per_sec']} img/s  "
                          f"train {train_content:.3f} + {train_style:.3f}  "
                          f"val {val_content:.3f} + {val_style:.3f} = {row['val_total']:.3f}"
                          f"{'  best' if is_best else ''}", flush=True)

                    interval_losses.zero_()
                    interval_steps, interval_start = 0, time.time()

                if step == total_steps:
                    break

    torch.save(state_dict_on_cpu(transformer), out / "last.model")
    print(f"done in {time.time() - start:.0f} s, best val loss {best_val:.3f}", flush=True)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--style", required=True, help="style image")
    parser.add_argument("--out", required=True, help="run directory")
    parser.add_argument("--style-size", type=int, default=512,
                        help="long side of the style image in px, 0 keeps the original size")
    parser.add_argument("--style-weight", type=float, default=1e5)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--images", type=int, default=40000, help="training images to pass through the network")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-every", type=int, default=250, help="validate every N steps")
    parser.add_argument("--init", help="start from this checkpoint instead of random weights")
    parser.add_argument("--images-dir", default="data/coco/val2017")
    parser.add_argument("--train-list", default=str(SPLITS / "train.txt"))
    parser.add_argument("--val-list", default=str(SPLITS / "val.txt"))
    parser.add_argument("--sample-list", default=str(SPLITS / "test.txt"))
    parser.add_argument("--samples", type=int, default=6, help="test images shown in samples/")
    parser.add_argument("--sample-size", type=int, default=384)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained-vgg", dest="pretrained_vgg", action="store_false",
                        help=argparse.SUPPRESS)  # random VGG weights, for the smoke test only
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
