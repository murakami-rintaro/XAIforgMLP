import timm
from PIL import Image
import torch
from torchvision import transforms
from torch.nn import functional as F
import timm.models.mlp_mixer
import numpy as np
import exchange_tensor_array as exchange
import matplotlib.pyplot as plt


#timmの学習済みモデル一覧を表示
#model_names = timm.list_models(pretrained=True)
#pprint(model_names)

#torch.device = torch.device("cpu")

#モデル作成
model = timm.create_model("gmlp_s16_224", pretrained=True)
print(type(model))

model.eval()


# 各層の重みはここから取得可能（みたい）
#print(type(model.state_dict()))
#print(model.state_dict().keys())
#for i in model.state_dict().values():
    #print(i.shape)


#model = timm.create_model("gmlp_s16_224", pretrained = True, features_only = True)
#model.eval()

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
#img = Image.open("mike.png") #入力画像
#print(type(img))
#print(img.size)

inputs = transform(img)
#print(type(inputs))
#print(inputs.type())
inputs = inputs.unsqueeze(0)
#print("input")
#print(inputs.shape, type(inputs))
#print(type(inputs[0][0][0][0]))


#ダミー入力の生成

inputs = [[[list(range(i * 224 , (i + 1) * 224)) for i in range(224)] for j in range(3)]]
inputs = np.array(inputs)
inputs = torch.as_tensor(inputs, dtype=torch.float)
print(type(inputs), inputs.shape, inputs.type())
print(inputs[0][0][0][0])
print(type(inputs[0][0][0][0]))
input_1 = inputs[0][0]
input_2 = inputs[0][1]
input_3 = inputs[0][2]
print("is same?")
print(torch.where(input_1 != input_2))
print(torch.where(input_2 != input_3))
print(torch.where(input_3 != input_1))



zeros = torch.zeros(1, 3, 224, 224, dtype=torch.float)
#推論
output = model(zeros)


block_inputs = []
block_outputs = []
#ブロックの入出力の抽出
for i in model.blocks:
    _input = i.block_input
    #_output = i.block_output
    block_inputs.append(_input)
    #block_outputs.append(_output[0])
    #print(type(_input), _input.shape)
    #print(type(_output), _output.shape)
#for i in block_inputs:
    #print(torch.cumsum(i, 1))
for i in block_inputs:
    tmp = exchange.exchange_tensor_to_array(i[0])
    print(type(tmp))
    #exchange.show_heatmap_with_colorbar(tmp)
    break
    #print(type(tmp), tmp.shape)
    tmp_output = model.pooling_only(i)
    print(type(tmp_output), tmp_output.shape)
exit()
"""
path = "zeros_input.txt"
with open(path, mode="w") as f:
    for i in block_inputs[0]:
        s = i.tolist()
        s = " ".join(map(str, s))
        f.write(s)
        f.write("\n")
"""
exit()



#print("is same?")
#for i in range(30 - 1):
    #print(torch.where(block_inputs[i + 1] != block_outputs[i]))
#print(type(model.blocks[0]))
#print(output.shape)
#print(model.forward(inputs).size()) #モデル全体の出力（のサイズ）
#print(model.forward_features(inputs).size()) #pooling前の出力（特徴量）（のサイズ）
#print(model.forward_features(inputs))
exit()
#print(output)
batch_probs = F.softmax(output, dim = 1)
batch_probs, batch_indices = batch_probs.sort(dim = 1, descending=True)
print(batch_probs[0][0])
print(batch_indices[0][0])