import timm
from PIL import Image
import torch
from torchvision import transforms
from torch.nn import functional as F
import timm.models.mlp_mixer
import numpy as np
import matplotlib.pyplot as plt
import shap

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
inputs = transform(img)
inputs = inputs.unsqueeze(0)
e = shap.DeepExplainer(model, inputs)
e = shap.ex
shap_values = e.shap_values(inputs)