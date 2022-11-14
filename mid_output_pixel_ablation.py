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

class MidOutputPixelAblation():
    """
    最後のblockを1要素ずつ0にした際のスコアの変化から各要素の寄与を算出
    """
    def __init__(self, model : timm.models.mlp_mixer.MlpMixer, border = 2) -> None:
        self.model = model
        self.inputs = []
        self.values = []
        self.get_classes()
        self.border = border
        self.x = None
        self.y = None
        self.base_class_index = None
        

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

        return (self.class_names[batch_indices[0][0]], batch_probs[0][0].item())

    
    def calc_value_input_image(self, _input : torch.Tensor, border = 2) -> np.array:
        """
        入力画像に対して、batch_size×batch_sizeのバッチを1つずつ0にした際のスコアの変化を予測への寄与とする
        寄与度は元々のスコアをx, (i,j)要素を0にした際のスコアをyijとしたとき、(yij - x) / xとする
        """
        #元々のスコアを取得
        base_class_name, base_prob = self.base_model_output(_input)
        #判定されたクラスのインデックスを取得
        self.base_class_index = self.class_names.index(base_class_name)
        
        #最初の中間層を取得
        self.block_output = self.model.blocks[0].block_output
        self.block_output_exchanged = exchange.exchange_tensor_to_array(self.block_output)

        value = np.zeros((N, N))
        self.p = []
        
        #最初の中間層から値の絶対値が閾値より大きいものの座標を取得
        self.y, self.x = np.where(abs(self.block_output_exchanged) > border)
        
        for i in self.y:
            for j in self.x:
                #(i,j)要素を保存した後0に
                r = _input[0, 0, i, j]
                b = _input[0, 1, i, j]
                g = _input[0, 2, i, j]
                _input[0, 0, i, j] = 0
                _input[0, 1, i, j] = 0
                _input[0, 2, i, j] = 0

                #(i,j)要素を削除した際のスコアを取得
                output = self.model(_input)
                batch_probs = F.softmax(output, dim=1)
                tmp_prob = batch_probs[0][self.base_class_index].item()
                self.p.append(tmp_prob)
                #tmp_value = (tmp_prob - base_prob) / base_prob * 100

        #スコアを記録するとともに入力を基に戻す
        for i in self.y:
            for j in self.x:
                value[i, j] = tmp_prob
                _input[0, 0, i, j] = r
                _input[0, 1, i, j] = b
                _input[0, 2, i, j] = g
                

        #self.inputs.append(_input)
        #self.values.append(value)
        return value