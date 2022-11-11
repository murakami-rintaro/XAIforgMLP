import timm
import exchange_tensor_array as exchange
import torch
from torch.nn import functional as F
from pathlib import Path
from torchvision.datasets.utils import download_url
import json
import numpy as np
import matplotlib.pyplot as plt

N = 224

class BatchlAblationCAMfrogMLP():
    """
    最後のblockを1要素ずつ0にした際のスコアの変化から各要素の寄与を算出
    """
    def __init__(self, model : timm.models.mlp_mixer.MlpMixer, batch_size = 16) -> None:
        self.model = model
        self.inputs = []
        self.values = []
        self.get_classes()
        self.batch_size = batch_size
        self.r = [ [0] * self.batch_size for _ in range(self.batch_size)]
        self.b = [ [0] * self.batch_size for _ in range(self.batch_size)]
        self.g = [ [0] * self.batch_size for _ in range(self.batch_size)]
        self.p = []

    def get_classes(self):
        if not Path("data/imagenet_class_index.json").exists():
            # ファイルが存在しない場合はダウンロードする。
            download_url("https://git.io/JebAs", "data", "imagenet_class_index.json")

        # クラス一覧を読み込む。
        with open("data/imagenet_class_index.json") as f:
            data = json.load(f)
            self.class_names = [x["ja"] for x in data]


    def base_model_output(self, _input : torch.Tensor) -> tuple:
        """画像をモデルに入力した際の一番スコアが高いラベルとそのスコアを返す"""
        base_output = self.model(_input)
        batch_probs = F.softmax(base_output, dim=1)
        batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)

        return (self.class_names[batch_indices[0][0]], batch_probs[0][0])

    def model_output(self, _input : torch.Tensor, class_name : str) -> float:
        """画像をモデルに入力した際の指定したラベルのスコアを返す"""
        base_output = self.model(_input)
        batch_probs = F.softmax(base_output, dim=1)
        batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)
        res = -1
        flag = False

        for probs, indices in zip(batch_probs, batch_indices):
            for k in range(1000):
                if self.class_names[indices[k]] == class_name:
                    res = probs[k]
                    flag = True
                    break
            if flag:
                break

        if res == -1:
            print("指定したクラス名は存在しません")

        return res

    
    def calc_value_input_image(self, _input : torch.Tensor, batch_size = 16) -> np.array:
        """
        入力画像に対して、batch_size×batch_sizeのバッチを1つずつ0にした際のスコアの変化を予測への寄与とする
        寄与度は元々のスコアをx, (i,j)要素を0にした際のスコアをyijとしたとき、(yij - x) / xとする
        """
        self.batch_size = batch_size
        #元々のスコアを取得
        base_class_name, base_prob = self.base_model_output(_input)
        #判定されたクラスのインデックスを取得
        base_class_index = self.class_names.index(base_class_name)

        value = np.zeros((N, N))
        self.r = [ [0] * self.batch_size for _ in range(self.batch_size)]
        self.b = [ [0] * self.batch_size for _ in range(self.batch_size)]
        self.g = [ [0] * self.batch_size for _ in range(self.batch_size)]
        self.p = []
        for i in range(0, N, self.batch_size):
            for j in range(0, N, self.batch_size):
                #(i,j)要素を保存した後0に
                for y in range(self.batch_size):
                    for x in range(self.batch_size):
                        u = i + y
                        v = j + x
                        self.r[y][x] = _input[0, 0, u, v].item()
                        self.b[y][x] = _input[0, 1, u, v].item()
                        self.g[y][x] = _input[0, 2, u, v].item()
                        _input[0, 0, u, v] = 0
                        _input[0, 1, u, v] = 0
                        _input[0, 2, u, v] = 0
                #plt.imshow(_input[0].permute(1,2,0))

                #(i,j)要素を削除した際のスコアを取得
                output = self.model(_input)
                batch_probs = F.softmax(output, dim=1)
                tmp_prob = batch_probs[0][base_class_index].item()
                self.p.append(tmp_prob)
                tmp_value = (tmp_prob - base_prob) / base_prob * 100
                
                #スコアを記録するとともに入力を基に戻す
                for y in range(self.batch_size):
                    for x in range(self.batch_size):
                        u = i + y
                        v = j + x
                        value[u, v] = tmp_value
                        _input[0, 0, u, v] += self.r[y][x]
                        _input[0, 1, u, v] += self.b[y][x]
                        _input[0, 2, u, v] += self.g[y][x]
                        

        #self.inputs.append(_input)
        #self.values.append(value)

        return value
