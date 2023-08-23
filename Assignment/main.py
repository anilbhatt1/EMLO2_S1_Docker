import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import requests
from PIL import Image
from io import BytesIO
import timm
import json

default_image_path = 'https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/test/12/image/image.jpg'

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--model-name', type=str, default='mobilenetv3_large_100', metavar='N',
                    help='Timm model to be used for inferencing')

parser.add_argument('--image-path', type=str, default=default_image_path, metavar='N',
                    help='Image path for inferencing')

if __name__ == '__main__':
    args = parser.parse_args()
    img_path = args.image_path
    if 'http' in img_path:
        image = Image.open(requests.get(img_path, stream=True).raw)
    else:
        image = Image.open(img_path)
    
    model_name = args.model_name
    model = timm.create_model(model_name, pretrained=True).eval()
    transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
    
    image_tensor = transform(image)
    output = model(image_tensor.unsqueeze(0))
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
    IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')
    preds = [{'label': IMAGENET_1k_LABELS[idx], 'value': round(val.item(), 3), 'idx':idx.item()} for val, idx in zip(values, indices)]
    print(f'Inferencing with {model_name}')
    print(f'Top Prediction : {json.dumps(preds[0])}')
    print(f'Top 5 Predictions : {json.dumps(preds)}')