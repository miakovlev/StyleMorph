"""Export a TransformerNet checkpoint to ONNX and check that ONNX Runtime matches PyTorch.

    python -m training.export_onnx runs/starry/best.model models/starry.onnx

The model takes a (1, 3, H, W) float tensor named "input" with pixels in 0-255;
H and W are free, the app pads them to a multiple of 4.
"""
import argparse
import warnings

import numpy as np
import onnxruntime as ort
import torch

from utils.transformer_net import TransformerNet

# Heights and widths the check runs on: square, landscape, portrait, tiny
CHECK_SIZES = [(256, 256), (384, 512), (512, 340), (48, 68)]


def load_transformer(checkpoint):
    model = TransformerNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()


def export(model, path, opset=17):
    dummy = torch.rand(1, 3, 256, 256) * 255
    dynamic = {0: "batch_size", 2: "height", 3: "width"}
    # The legacy TorchScript exporter is deprecated, but the new one needs onnxscript, which needs numpy 2,
    # and the app's onnxruntime 1.18 is pinned to numpy 1. Its tracer warnings are expected for this model.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (dummy,), path, input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": dynamic, "output": dynamic}, opset_version=opset, dynamo=False)


def max_difference(model, path, sizes=CHECK_SIZES, seed=0):
    """Largest absolute difference between PyTorch and ONNX Runtime outputs, on the 0-255 scale."""
    session = ort.InferenceSession(path)
    rng = np.random.default_rng(seed)
    differences = []
    for height, width in sizes:
        x = rng.uniform(0, 255, (1, 3, height, width)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        actual = session.run(None, {"input": x})[0]
        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        differences.append(float(np.abs(actual - expected).max()))
    return max(differences)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="TransformerNet state_dict (.model)")
    parser.add_argument("output", help="ONNX file to write")
    parser.add_argument("--tolerance", type=float, default=0.01, help="max allowed difference on the 0-255 scale")
    args = parser.parse_args(argv)

    model = load_transformer(args.checkpoint)
    export(model, args.output)
    difference = max_difference(model, args.output)
    print(f"{args.output}: max |PyTorch - ONNX| = {difference:.5f} on sizes {CHECK_SIZES}")
    if difference > args.tolerance:
        raise SystemExit(f"difference {difference:.5f} is above the tolerance {args.tolerance}")


if __name__ == "__main__":
    main()
