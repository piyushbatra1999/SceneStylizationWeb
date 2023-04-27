import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time

def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

vgg_path ='./networks/vgg.pth'
decoder_path ='networks/decoder.pth'
Trans_path = 'networks/transformer.pth'
embedding_path = 'networks/embedding.pth'
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path='output'
preserve_color='store_true'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

def device_info():
    return device


vgg = StyTR.vgg
vgg.load_state_dict(torch.load(vgg_path))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
Trans = transformer.Transformer()
embedding = StyTR.PatchEmbed()

decoder.eval()
Trans.eval()
vgg.eval()
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(decoder_path)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(Trans_path)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(embedding_path)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

network = StyTR.StyTrans(vgg,decoder,embedding,Trans)
network.eval()
network.to(device)
print('Network Loaded!!')

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)   
content_tf1 = content_transform()   
   
def style_transfer(content_path,style_path,content_size=512,style_size=512,crop='store_true'):
    print(f'Transferring style from {style_path} to {content_path}')    
    content = content_tf(Image.open(content_path).convert("RGB"))

    h,w,c=np.shape(content)    
    style_tf1 = style_transform(h,w)
    style = style_tf(Image.open(style_path).convert("RGB"))

    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)

    with torch.no_grad():
        output= network(content,style)[0]
    output = output.cpu()
    print(f'Style transferred!!')
    save_image(output, 'output/output.jpg')