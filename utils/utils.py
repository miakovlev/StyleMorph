import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps


def load_image(source, max_size=512, size=None, scale=None):
    img = source if isinstance(source, Image.Image) else Image.open(source)
    # Phone photos keep their rotation in EXIF; apply it so the image is upright
    img = ImageOps.exif_transpose(img).convert('RGB')

    if max(img.size) > max_size:
        scaling_factor = max_size / float(max(img.size))
        new_size = tuple([int(dim * scaling_factor) for dim in img.size])
        img = img.resize(new_size, Image.LANCZOS)

    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.LANCZOS)

    return img


def stylize(content_image, model_path):
    session = ort.InferenceSession(model_path)

    content_image = load_image(content_image)
    content_image = np.array(content_image).transpose(2, 0, 1).astype(np.float32)
    content_image = np.expand_dims(content_image, axis=0)

    # The network downsamples by 4, so pad to a multiple of 4 and crop back to keep the input size
    height, width = content_image.shape[2:]
    padding = ((0, 0), (0, 0), (0, -height % 4), (0, -width % 4))
    content_image = np.pad(content_image, padding, mode="reflect")

    outputs = session.run(None, {"input": content_image})
    output = outputs[0][:, :, :height, :width]
    return output


def image_preprocess(image):
    image = image.squeeze(0).clip(0, 255).astype("uint8")
    image = image.transpose(1, 2, 0)
    image = Image.fromarray(image)
    return image


def resize_image_proportionally(image, max_size):
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(max_size * height / width)
    else:
        new_height = max_size
        new_width = int(max_size * width / height)
    return image.resize((new_width, new_height), Image.LANCZOS)
