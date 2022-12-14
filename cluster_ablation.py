import timm
from PIL import Image
import torch
from torchvision import transforms
from torch.nn import functional as F
import timm.models.mlp_mixer
import numpy as np
import exchange_tensor_array as exchange
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision.datasets.utils import download_url
import json

from scipy.spatial import Delaunay
from collections import defaultdict
from math import factorial
from collections import defaultdict
import bisect

from sklearn.cluster import DBSCAN, KMeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import copy

class cluster_ablation():
    """
    最初の中間層出力をクラスタリングして、クラスタ毎の凸法をマスクとしてshapley値を算出
    """
    
    def __init__(self, model : timm.models.mlp_mixer.MlpMixer) -> None:
        """初期化"""
        self.model = model
        self.class_names = None
        self.get_classes()
        self.inputs = []
        self.block_output_exchanged = []
        self.trasnform_for_model = transforms.Compose([
            transforms.Resize(256),  # (256, 256) で切り抜く。
            transforms.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
            transforms.ToTensor(),  # テンソルにする。
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 標準化する。
        ])
        self.trasnform_for_result = transforms.Compose([
            transforms.Resize(256),  # (256, 256) で切り抜く。
            transforms.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
        ])
        self.border = 300
        self.min_samples = 10
    def get_classes(self):
        """クラス一覧の名前を日本語で取得"""
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

    def in_hull_(self, point : list, hull : list):
        """点群hullからなる凸包内に点pointが入っているかを判定"""
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(point) >= 0
    
    def meguru_bisect_for_filitering(self, mid_output : np.ndarray, border = 300) -> float:
        """中間層の出力をフィルタリングする閾値をピックアップする特徴量の数を基に二分探索で求める"""
        ok = 0
        ng = max(abs(mid_output.max()), abs(mid_output.min())) + 1
        while abs(ok - ng) > 0.01:
            mid = (ok + ng) / 2
            if len(np.where(abs(mid_output) > mid)[0]) >= border:
                ok = mid
            else:
                ng = mid
        return ok


    def calc_shapley(self, img : torch.Tensor):

        """1枚の入力画像に対するshapley値を算出"""

        #元々のスコアを取得
        class_name, base_prob = self.base_model_output(img)
        #判定されたクラスのインデックスを取得
        class_index = self.class_names.index(class_name)

        #最初の中間層の出力を取得
        mid_0 = exchange.exchange_tensor_to_array(self.model.blocks[0].block_output)
        self.block_output_exchanged.append(mid_0)

        #最初の中間層をフィルタリング
        filtering_border = self.meguru_bisect_for_filitering(mid_output=mid_0, border=300)
        y, x = np.where(abs(mid_0) > filtering_border)
        p = [ [y[i], x[i]] for i in range(len(y))]
        #DBSCANでクラスタリング(パラメータは別途求めること)
        db = DBSCAN(eps=9, min_samples=5)
        pred = db.fit_predict(p)

        #クラスタ数
        n_c = len(set(pred))

        #shaply値格納
        shap = [0] * n_c
        #各マスクされた画像に対するスコア格納
        probs = [0] * pow(2, n_c)
        #マスクする際のバックアップ
        input_backup = defaultdict(lambda : [])

        #クラスタごとに点群を求める。各点群の座標の最大値最小値も記録しておく
        clusters = defaultdict(lambda : [])
        left = defaultdict(lambda : 1000)
        right = defaultdict(lambda : -1000)
        up = defaultdict(lambda : 1000)
        down = defaultdict(lambda : -1000)

        for v, u, c in zip(y, x, pred):
            if c == -1:
                continue
            clusters[c].append((v, u))
            left[c] = min(left[c], u)
            right[c] = max(right[c], u)
            up[c] = min(up[c], v)
            down[c] = max(down[c], v)

        #各画素ごとにどのクラスタの凸包に属するかを判定
        masks = defaultdict(lambda : [])
        mask_flag = [ [True] * 224 for _ in range(224)]

        for c in range(n_c - 1):
            for i in range(up[c], down[c] + 1):
                for j in range(left[c], right[c] + 1):
                    if self.in_hull_([i, j], clusters[c]):
                        masks[c].append((i, j))
                        mask_flag[i][j] = False
        for i in range(224):
            for j in range(224):
                if mask_flag[i][j]:
                    masks[n_c - 1].append((i, j))

        #マスクした画像保存用
        tmp_inputs = []

        values = []

        #全組み合わせを総当たり
        for i in range(2 ** n_c):
            input_backup = defaultdict(lambda : [])
            for c in range(n_c):
                if i >> c & 1:
                    #マスクする
                    for y, x in masks[c]:
                        r, b, g = img[0, 0, y, x].item(), img[0, 1, y, x].item(), img[0, 2, y, x].item()
                        input_backup[(y, x)] = [r, b, g]
                        img[0, 0, y, x] = 0
                        img[0, 1, y, x] = 0
                        img[0, 2, y, x] = 0

            
            #モデルに流してスコアを得る
            tmp_output = self.model(img)
            tmp_batch_probs = F.softmax(tmp_output, dim=1)
            tmp_prob = tmp_batch_probs[0][class_index].item()
            
            probs[i] = tmp_prob
            values.append(tmp_prob)
            
            tmp_inputs.append(copy.deepcopy(img))
            
            #入力画像を元に戻す
            for key, val in input_backup.items():
                v, u = key
                r, b, g = val
                img[0, 0, v, u] = r
                img[0, 1, v, u] = b
                img[0, 2, v, u] = g
        
        #各マスクした画像に対するスコアを基にクラスタごとのshapley値を算出
        
        for i in range(2 ** n_c):
            count = 0
            flags = []
            for j in range(n_c):
                if i >> j & 1:
                    count += 1
                else:
                    flags.append(j)
            for j in flags:
                tmp = factorial(count) * factorial(n_c - count - 1) * (-probs[i + pow(2, j)] + probs[i]) / factorial(n_c)
                shap[j] += tmp
        
        #return shap, tmp_inputs, values, masks, p, pred, mask_flag, clusters, left, right, up, down
        return shap, p, pred


    def calc_shapley_save_img(self, input_path : str, output_path : str, mode = "D", border=300, eps=9, min_samples = 10, n_c_max = 10):
        """1枚の入力画像に対するshapley値を算出.クラスタごとにshapley値に応じて元画像にマッピングする."""
        
        #パラメータ更新
        self.border = border
        self.eps = eps
        self.min_samples = min_samples
        self.n_c_max = n_c_max
        
        
        #pathから入力画像を読み込んで変換
        _input = Image.open(input_path)
        img = self.trasnform_for_model(_input)
        img = img.unsqueeze(0)
    
        #元々のスコアを取得
        class_name, base_prob = self.base_model_output(img)
        #判定されたクラスのインデックスを取得
        class_index = self.class_names.index(class_name)

        #最初の中間層の出力を取得
        mid_0 = exchange.exchange_tensor_to_array(self.model.blocks[0].block_output)
        self.block_output_exchanged.append(mid_0)

        #最初の中間層をフィルタリング
        filtering_border = self.meguru_bisect_for_filitering(mid_output=mid_0, border=self.border)
        y, x = np.where(abs(mid_0) > filtering_border)
        p = [ [y[i], x[i]] for i in range(len(y))]
        
        """
        #DBSCANでクラスタリング(パラメータは別途求めること)
        db = DBSCAN(eps=9, min_samples=5)
        pred = db.fit_predict(p)
        """
        
        if mode == "D":
            #DBSCANでクラスタリング(クラスタ数がn_c_max未満になるようにmin_samplesを調整)
            
            self.eps = 5
            self.outlier = self.border
            self.n_c = 0
            self.pred = np.zeros(1)
            paras = defaultdict(lambda : [])
            for eps in range(3, 30):
                for min_samples in (4, 20):
                    db = DBSCAN(eps = eps, min_samples=min_samples)
                    pred = db.fit_predict(p)
                    n_c = len(set(pred))
                    
                    #各クラスタが4つ以上の点があるかを確認する(なぜか1つでクラスタリングされる事例があったので)
                    flag = True 
                    for i in list(set(pred)):
                        if i == -1:
                            continue
                        if np.count_nonzero(pred == i) < 4:
                            flag = False
                            break
                    if not flag:
                        continue
                    outlier = np.count_nonzero(pred==-1)
                    #if n_c == self.n_c_max: #クラス数が閾値以下で最大かどうか
                    paras[n_c].append([outlier, pred])
            
            #fig, ax = plt.subplots()
            # plt.scatter(x, y,s=100,c=self.pred, cmap="Blues")
            # plt.colorbar()
            # plt.xlim(0, 223)
            # plt.ylim(223, 0)
            
            para_index = bisect.bisect_left(sorted(paras.keys()), self.n_c_max)#クラスタ数が閾値以下で最大のもの
            if para_index == len(paras.keys()):#クラスタ数が閾値以下のものが存在しない時
                para_index -= 1
            n_c = sorted(paras.keys())[para_index]
            print("n_c list : {}".format(paras.keys()))
            print("n_c={}".format(n_c))
            print(sorted(paras[n_c]))
            pred = sorted(paras[n_c])[0][1] #外れ値が最も少ないもの
            
            
            #pred = self.pred
            #n_c = self.n_c
            #return pred, paras
            
            # db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            # pred = db.fit_predict(p)
            # #クラスタ数
            # n_c = len(set(pred))            
            # while n_c <= 3:
            #     self.min_samples -= 1
            #     db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            #     pred = db.fit_predict(p)
            #     n_c = len(set(pred))

            # while n_c >= self.n_c_max:
            #     db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            #     pred = db.fit_predict(p)
            #     #クラスタ数
            #     n_c = len(set(pred))
            #     self.eps += 1
        elif mode == "K":
            #kmeansでクラスタリング
            km = KMeans(n_clusters=7, random_state=0)
            pred = km.fit_predict(p)
            n_c = len(set(pred)) + 1
        elif mode=="X":
            #X-meansでクラスタリング
            tmp_p = np.array([[y[i], x[i]] for i in range(len(y))])
            xm_i = xmeans(data=tmp_p, k_max=10, ccore=True)
            xm_i.process()
            pred = [0] * len(y)
            n_c = len(xm_i._xmeans__clusters) + 1
            for i in range(n_c - 1):
                for j in xm_i._xmeans__clusters[i]:
                    pred[j] = int(i)

        #shaply値格納
        shap = [0] * n_c
        #各マスクされた画像に対するスコア格納
        probs = [0] * pow(2, n_c)
        #マスクする際のバックアップ
        input_backup = defaultdict(lambda : [])

        #クラスタごとに点群を求める。各点群の座標の最大値最小値も記録しておく
        clusters = defaultdict(lambda : [])
        left = defaultdict(lambda : 1000)
        right = defaultdict(lambda : -1000)
        up = defaultdict(lambda : 1000)
        down = defaultdict(lambda : -1000)

        for v, u, c in zip(y, x, pred):
            if c == -1:
                continue
            clusters[c].append((v, u))
            left[c] = min(left[c], u)
            right[c] = max(right[c], u)
            up[c] = min(up[c], v)
            down[c] = max(down[c], v)

        #各画素ごとにどのクラスタの凸包に属するかを判定
        masks = defaultdict(lambda : [])
        mask_flag = [ [True] * 224 for _ in range(224)]

        if mode != "D":
            n_c += 1
        flag_count = 0
        for c in range(n_c - 1):
            for i in range(up[c], down[c] + 1):
                for j in range(left[c], right[c] + 1):
                    if self.in_hull_([i, j], clusters[c]):
                        masks[c].append((i, j))
                        mask_flag[i][j] = False
                        flag_count += 1
            print(flag_count)
        for i in range(224):
            for j in range(224):
                if mask_flag[i][j]:
                    masks[n_c - 1].append((i, j))
        if mode != "D":
            n_c -= 1

        #マスクした画像保存用
        tmp_inputs = []

        values = []

        #全組み合わせを総当たり
        for i in range(2 ** n_c):
            input_backup = defaultdict(lambda : [])
            for c in range(n_c):
                if i >> c & 1:
                    #マスクする
                    for y, x in masks[c]:
                        r, b, g = img[0, 0, y, x].item(), img[0, 1, y, x].item(), img[0, 2, y, x].item()
                        input_backup[(y, x)] = [r, b, g]
                        img[0, 0, y, x] = 0
                        img[0, 1, y, x] = 0
                        img[0, 2, y, x] = 0
            #print(len(np.where(img[0][0]==len(masks[c]))))

            
            #モデルに流してスコアを得る
            tmp_output = self.model(img)
            tmp_batch_probs = F.softmax(tmp_output, dim=1)
            tmp_prob = tmp_batch_probs[0][class_index].item()
            
            probs[i] = tmp_prob
            values.append(tmp_prob)
            
            tmp_inputs.append(copy.deepcopy(img))
            
            #入力画像を元に戻す
            for key, val in input_backup.items():
                v, u = key
                r, b, g = val
                img[0, 0, v, u] = r
                img[0, 1, v, u] = b
                img[0, 2, v, u] = g
        return probs, tmp_inputs, masks
        #各マスクした画像に対するスコアを基にクラスタごとのshapley値を算出
        
        for i in range(2 ** n_c):
            count = 0
            flags = []
            for j in range(n_c):
                if i >> j & 1:
                    count += 1
                else:
                    flags.append(j)
            for j in flags:
                tmp = factorial(count) * factorial(n_c - count - 1) * (-probs[i + pow(2, j)] + probs[i]) / factorial(n_c)
                shap[j] += tmp
        
        
        
        #クラスタごとにマッピング.shapley値が最も大きいクラスタを255とする
        cover_img = np.zeros((224, 224, 4), dtype=np.uint8)
        for i in range(224):
            for j in range(224):
                cover_img[i, j, 3] = 0
        max_shap = max(shap[:-1])
        for i, c in zip(p, pred):
            if c == -1 or shap[c] < max_shap / 2:
                continue
            y, x = i
            cover_img[y, x, 0] = 255 * shap[c] / max_shap
            cover_img[y, x, 3] = 255
        cover_img = Image.fromarray(cover_img)
        base_img = self.trasnform_for_result(_input)
        base_img.paste(cover_img, (0, 0), cover_img)
        base_img.save(output_path, quality=95)
        
        
        
        
        #return shap, tmp_inputs, values, masks, p, pred, mask_flag, clusters, left, right, up, down
        return shap, p, pred