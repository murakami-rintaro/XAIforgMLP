import torch
import numpy as np
import matplotlib.pyplot as plt


def exchange_tensor_to_array(t : torch.Tensor) -> np.ndarray:
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
            res[y][x] = t[0, i, j]
    return res

def show_heatmap_with_colorbar(x : np.ndarray) -> None:
    """numpy行列のヒートマップを出力"""
    _, ax = plt.subplots()
    im = ax.imshow(x)
    plt.colorbar(im)
    plt.show()


def rescaling_to_first_array(x: list[np.ndarray]) -> list[np.ndarray]:
    up = x[0].max()
    bottom = x[0].min()
    _range = up - bottom
    for i in x[1:]:
        tmp_up = i.max()
        tmp_bottom = i.min()
        tmp_range = tmp_up - tmp_bottom
        tmp_rate = _range / tmp_range
        i -= tmp_bottom
        i *= tmp_rate
        i += tmp_bottom * tmp_rate
    return x
        