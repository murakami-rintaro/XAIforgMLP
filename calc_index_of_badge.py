import timm
import torch
import timm.models.mlp_mixer
import numpy as np

#モデル作成
model = timm.create_model("gmlp_s16_224", pretrained=True)
model.eval()

"""
# 0埋め時の最初のモジュールの出力をテキストファイルから読み込み
path = "zeros_input.txt"
zeros_input = []
with open(path, mode="r") as f:
    for s in f:
        s.strip()
        s = s.split()
        s = list(map(float, s))
        zeros_input.append(s)
zeros_input = np.array(zeros_input)
zeros_input = torch.as_tensor(zeros_input, dtype=torch.float)
#print(zeros_input.shape)
"""

zeros = torch.zeros(1, 3, 224, 224, dtype=torch.float)

output = model(zeros)
for b in model.blocks:
    zeros_input = b.block_input[0]
    break

#true_zeros_input = model.model_input
#print(true_zeros_input)
#print(true_zeros_input.shape)


counts = [[0] * 224 for i in range(224)]
for i in range(0, 1, 16):
    for j in range(0, 224, 16):
        # ダミー入力の生成
        almost_zeros = np.zeros((1, 3, 224, 224))
        almost_zeros[0][0][i][j] = 1
        almost_zeros[0][1][i][j] = 1
        almost_zeros[0][2][i][j] = 1
        almost_zeros = torch.as_tensor(almost_zeros, dtype=torch.float)

        # 推論
        output = model(almost_zeros)
        true_almost_zeros_input = model.model_input

        for b in model.blocks:
            almost_zeros_input = b.block_input[0]
            break
        #print(almost_zeros_input.shape)
        #print(almost_zeros_input)
        #print(zeros_input)
        diff_points = torch.where(zeros_input != almost_zeros_input)
        #true_diff_points0 = torch.where(true_zeros_input[0][0] != true_almost_zeros_input[0][0])
        #print("i = {}, j = {}".format(i, j))
        #print(true_diff_points0)
        #print(type(diff_points))
        print("i = {}, j = {}".format(i, j))
        print(diff_points)
        #print(diff_points[0].shape, diff_points[1].shape)
        

