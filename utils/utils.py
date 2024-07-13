import onnxruntime as ort
import numpy as np
from PIL import Image


def load_image(filename, max_size=512, size=None, scale=None):
    img = Image.open(filename).convert('RGB')

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

    outputs = session.run(None, {"input": content_image})
    output = outputs[0]
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
