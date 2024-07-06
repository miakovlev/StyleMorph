import torch
from torchvision import transforms
import re
from utils.transformer_net import TransformerNet
from PIL import Image


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def stylize(content_image, model_path):
    device = torch.device("cpu")

    content_image = load_image(content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model_path, map_location=device)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
    return output[0]


def image_preprocess(image):
    image = image.clone().clamp(0, 255).numpy()
    image = image.transpose(1, 2, 0).astype("uint8")
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
