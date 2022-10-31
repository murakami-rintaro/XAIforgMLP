import timm
from PIL import Image
import torch
from torchvision import transforms
from torch.nn import functional as F
import timm.models.mlp_mixer
import numpy as np
import exchange_tensor_array as exchange
import matplotlib.pyplot as plt

#モデル作成
model = timm.create_model("gmlp_s16_224", pretrained=True)
model.eval()


transform = transforms.Compose(
    [
        transforms.Resize(256),  # (256, 256) で切り抜く。
        transforms.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
        transforms.ToTensor(),  # テンソルにする。
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 標準化する。
    ]
)


img = Image.open("cat.jpg")
exchange.show_heatmap_with_colorbar(img)
inputs = transform(img)
inputs = inputs.unsqueeze(0)


#推論
output = model(inputs)

block_inputs = []
block_outputs = []
#ブロックの入出力の抽出
for i in model.blocks:
    _input = i.block_input
    _output = i.block_output
    block_inputs.append(_input)
    block_outputs.append(_output)

for i in block_inputs:
    tmp = exchange.exchange_tensor_to_array(i[0])
    #print(type(tmp))
    exchange.show_heatmap_with_colorbar(tmp)
    #print(type(tmp), tmp.shape)
    #tmp_output = model.pooling_only(i)
    #print(type(tmp_output), tmp_output.shape)


for i in block_outputs:
    tmp = exchange.exchange_tensor_to_array(i[0])
    #print(type(tmp))
    exchange.show_heatmap_with_colorbar(tmp)