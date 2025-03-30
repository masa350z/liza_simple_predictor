# modules/dataset.py
"""学習データセット作成モジュール

   * k個の過去データを特徴量とし、pr_k(将来)後の価格が上昇か下降かをラベルとする。
   * クラスバランスを取るため、UPとDOWNが同じくらいになるようにする。
   * 学習/検証/テストに分割する。
   * 入力の正規化やサンプリングなどを行う。
"""

import numpy as np


def create_dataset(prices,
                   k,
                   future_k,
                   train_ratio=0.6,
                   valid_ratio=0.2,
                   down_sampling=1,
                   normalize=True,
                   split=True,
                   balance_class=True):
    """特徴量とラベルを生成してtrain/valid/testに分割

    概要:
        ・連続する時系列データを使い、(k個の価格) -> (future_k後の上昇or下降) を2値分類する
        ・UP=1, DOWN=0 の2クラスを(1,0) / (0,1)のone-hotに変換
        ・クラスバランスのため、UPとDOWNを同数になるように整列
        ・学習/バリデーション/テストの3分割

    Args:
        prices (list of float): 時系列の価格データ
        k (int): 過去何個分の価格を特徴量とするか
        future_k (int): 何個先の価格を見るか
        train_ratio (float): 学習データセットの比率
        valid_ratio (float): バリデーションデータセットの比率
        down_sampling (int): データ間引き（1なら全て、2なら半分、など）

    Returns:
        tuple: ( (train_x, train_y), (valid_x, valid_y), (test_x, test_y) )
    """
    # 1. 時系列を numpy配列に変換
    prices_arr = np.array(prices, dtype="float32")

    # 2. (k + future_k)個ずつ取り出せるだけスライスして2次元化
    #    [ [p0..p(k-1)], [p1..p(k)], ..., [pN..p(N+k-1)], ... ]
    data_list = []
    label_list = []
    max_index = len(prices_arr) - (k + future_k)

    for start_idx in range(max_index):
        x_slice = prices_arr[start_idx: start_idx + k]
        future_price_now = prices_arr[start_idx + k - 1]
        future_price_then = prices_arr[start_idx + k + future_k - 1]
        # 上昇 or 下降をラベル
        is_up = 1 if future_price_then > future_price_now else 0
        # one-hot化
        if is_up == 1:
            y = [1, 0]
        else:
            y = [0, 1]

        data_list.append(x_slice)
        label_list.append(y)

    data_array = np.array(data_list, dtype="float32")
    label_array = np.array(label_list, dtype="int32")

    # 3. Down Sampling
    data_array = data_array[::down_sampling]
    label_array = label_array[::down_sampling]

    if balance_class:
        # 4. クラスバランスを取る (UP=1とDOWN=1が同数になるようにサンプリング)
        data_array, label_array = _balance_up_down(data_array, label_array)
    
    if normalize:
        # 5. 正規化 (最大値-最小値で割る)
        data_array = _minmax_scale(data_array)
        data_array = data_array.astype('float16')
    
    if split:
        # 6. 学習/検証/テストに分割
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = _split_data(
            data_array, label_array, train_ratio, valid_ratio
        )

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    else:
        return data_array, label_array


def _balance_up_down(data_array, label_array):
    """UP / DOWN クラス数を揃える"""
    # label_array: shape (N, 2)  [1,0] -> UP, [0,1] -> DOWN
    up_indices = np.where(label_array[:, 0] == 1)[0]
    dn_indices = np.where(label_array[:, 1] == 1)[0]

    up_count = len(up_indices)
    dn_count = len(dn_indices)

    if up_count == 0 or dn_count == 0:
        return data_array, label_array  # クラスがどちらかに偏りすぎた場合は何もせず返す

    min_count = min(up_count, dn_count)
    np.random.shuffle(up_indices)
    np.random.shuffle(dn_indices)

    up_indices = up_indices[:min_count]
    dn_indices = dn_indices[:min_count]

    balanced_indices = np.concatenate([up_indices, dn_indices])
    np.random.shuffle(balanced_indices)

    data_array = data_array[balanced_indices]
    label_array = label_array[balanced_indices]

    return data_array, label_array


def _minmax_scale(data_2d):
    """2次元配列に対して (要素 - min) / (max - min) を行う"""
    # data_2d shape : (N, k)
    max_vals = data_2d.max(axis=1, keepdims=True)
    min_vals = data_2d.min(axis=1, keepdims=True)
    scaled = (data_2d - min_vals) / (max_vals - min_vals + 1e-8)
    return scaled


def _split_data(x_data, y_data, train_ratio, valid_ratio):
    """train/valid/test の3分割を行う"""
    total_len = len(x_data)
    train_len = int(total_len * train_ratio)
    valid_len = int(total_len * valid_ratio)

    train_x = x_data[:train_len]
    train_y = y_data[:train_len]

    valid_x = x_data[train_len: train_len + valid_len]
    valid_y = y_data[train_len: train_len + valid_len]

    test_x = x_data[train_len + valid_len:]
    test_y = y_data[train_len + valid_len:]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)