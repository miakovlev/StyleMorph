"""Split a folder of images into train.txt and val.txt, leaving out the hand-picked test.txt.

    python -m training.make_splits data/coco/val2017 training/splits/coco_val2017 --val 490

The lists are committed so that every run and every machine trains and validates on the same images.
"""
import argparse
import random
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def make_splits(images_dir, splits_dir, val_size, seed=42):
    splits_dir = Path(splits_dir)
    test_list = splits_dir / "test.txt"
    test = set(test_list.read_text().split()) if test_list.exists() else set()

    names = sorted(path.name for path in Path(images_dir).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    missing = test - set(names)
    if missing:
        raise SystemExit(f"test images not found in {images_dir}: {sorted(missing)}")

    rest = [name for name in names if name not in test]
    random.Random(seed).shuffle(rest)
    splits = {"train": sorted(rest[val_size:]), "val": sorted(rest[:val_size])}
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, split_names in splits.items():
        (splits_dir / f"{split}.txt").write_text("\n".join(split_names) + "\n")
    return {split: len(split_names) for split, split_names in splits.items()} | {"test": len(test)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir")
    parser.add_argument("splits_dir")
    parser.add_argument("--val", type=int, default=490, help="validation images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(make_splits(args.images_dir, args.splits_dir, args.val, args.seed))


if __name__ == "__main__":
    main()
