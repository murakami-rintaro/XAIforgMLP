import torch
import numpy as np
import matplotlib.pyplot as plt


def exchange_tensor_to_array(t : torch.Tensor) -> np.array:
    """
    blockの入出力をモデルの入力画像の形状に変換
    """
    res = np.zeros((224, 224))
    for i in range(196):
        budge_y = (i // 14) * 16
        budge_x =(i % 14) * 16
        for j in range(256):
            dy = j // 16
            dx = j % 16
            y = budge_y + dy
            x = budge_x + dx
            res[y, x] = t[0, i, j]
    return res

def show_heatmap_with_colorbar(x : np.ndarray) -> None:
    """numpy行列のヒートマップを出力"""
    _, ax = plt.subplots()
    im = ax.imshow(x)
    plt.colorbar(im)
    plt.show()
    
