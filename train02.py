# %%
from modules.trainer import Trainer
from technical_funcs import calc_sma, calc_bollinger_bands, calc_macd, calc_rsi
from tensorflow.keras import layers, Model, Input
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from modules.data_loader import load_csv_data
from modules.models import build_hybrid_technical_model

# 必要な分だけGPUメモリを確保する
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def normalize_by_price_window(data, rsi_idx=11, eps=1e-8):
    """
    破壊的に正規化を行う（in-place）
    data : np.ndarray, shape = (N, k, 12)
           チャネル構成:
             0   : price
             1-10: テクニカル（差分）
             11  : RSI (0〜100)
    返値 : 正規化された同じ data (in-place 処理)
    """
    prices = data[:, :, 0]                           # (N, k)
    mean_p = np.mean(prices, axis=1, keepdims=True)  # (N, 1)
    std_p = np.std(prices, axis=1, keepdims=True) + eps  # (N, 1)

    # z-score normalize for price
    data[:, :, 0] = (prices - mean_p) / std_p

    # normalize diff indicators (sma, boll, macd)
    data[:, :, 1:rsi_idx] /= std_p[:, :, None]       # (N, k, 10)

    # normalize RSI to [-1, 1]
    data[:, :, rsi_idx] = (data[:, :, rsi_idx] - 50.0) / 50.0

    return data


def build_dataset_with_normalization_in_batches(input_array,
                                                k,
                                                p,
                                                batch_size=10000,
                                                rsi_idx=11,
                                                eps=1e-8):
    """
    ミニバッチで (N, k, 12) の正規化済みデータとラベルを生成（出力は float16）

    input_array: (T, 12) 価格 + テクニカル指標（RSIはインデックス11に想定）
    k: 入力系列長
    p: future予測系列差分
    batch_size: スライス単位の処理件数
    rsi_idx: RSIのチャネル位置（デフォルト: 11）
    eps: ゼロ割回避定数

    戻り値:
        data_array: (N, k, 12) np.ndarray (float16, 正規化済み)
        label_array: (N, 2)    np.ndarray (float16, 上昇/下降 one-hot)
    """
    T = len(input_array)
    max_index = T - (k + p)
    data_batches = []
    label_batches = []

    for start in range(0, max_index, batch_size):
        end = min(start + batch_size, max_index)
        B = end - start

        batch_data = np.zeros((B, k, input_array.shape[1]), dtype=np.float32)
        batch_label = np.zeros((B, 2), dtype=np.float16)  # ✅ float16に変更

        for i, idx in enumerate(range(start, end)):
            x_slice = input_array[idx:idx + k + p]
            current_price = x_slice[k - 1, 0]
            future_price = x_slice[-1, 0]
            is_up = 1.0 if future_price > current_price else 0.0
            batch_label[i] = [is_up, 1.0 - is_up]

            batch_data[i] = x_slice[:k]

        # 正規化 (in-place)
        prices = batch_data[:, :, 0]                             # (B, k)
        mean_p = np.mean(prices, axis=1, keepdims=True)          # (B, 1)
        std_p = np.std(prices, axis=1, keepdims=True) + eps      # (B, 1)

        batch_data[:, :, 0] = (prices - mean_p) / std_p
        batch_data[:, :, 1:rsi_idx] /= std_p[:, :, None]
        batch_data[:, :, rsi_idx] = (batch_data[:, :, rsi_idx] - 50.0) / 50.0

        # float16化して保持
        data_batches.append(batch_data.astype(np.float16))
        label_batches.append(batch_label)

    data_array = np.concatenate(data_batches, axis=0)
    label_array = np.concatenate(label_batches, axis=0)
    return data_array, label_array


# %%
pair = "BTCJPY"
csv_file_name = f"sample_{pair}_1m.csv"
csv_file_path = os.path.join("data", csv_file_name)
_, prices_ = load_csv_data(csv_file_path, skip=1)

prices = np.array(prices_, dtype=np.float32)

k = 360
p = 120

sma_short_window = 20
sma_mid_window = 60
sma_long_window = 120

bollinger_band_window = 20
bollinger_band_sigma = 2.0


macd_short_window = 12
macd_long_window = 26
macd_signal_window = 9

rsi_window = 14

# 1) SMA
sma_short = calc_sma(prices, sma_short_window)
sma_mid = calc_sma(prices, sma_mid_window)
sma_long = calc_sma(prices, sma_long_window)

# 2) Bollinger Bands
bollinger_band_center, bollinger_band_upper, bollinger_band_lower = calc_bollinger_bands(
    prices, bollinger_band_window, bollinger_band_sigma)

# 3) MACD
macd_line, signal_line = calc_macd(
    prices, short_window=macd_short_window, long_window=macd_long_window, signal_window=macd_signal_window)

# 4) RSI
rsi = calc_rsi(prices, window=rsi_window)

bollinger_bands = np.stack(
    [bollinger_band_center, bollinger_band_upper, bollinger_band_lower], axis=1)
macds = np.stack([macd_line, signal_line], axis=1)

short_mid = sma_short - sma_mid
short_long = sma_short - sma_long
mid_short = sma_mid - sma_short
mid_long = sma_mid - sma_long
long_short = sma_long - sma_short
long_mid = sma_long - sma_mid
smas = np.stack([short_mid, short_long,
                 mid_short, mid_long,
                 long_short, long_mid], axis=1)

bollinger_bands_input = bollinger_bands - prices.reshape(-1, 1)
macds_input = macds[:, 0] - macds[:, 1]

input_array = np.concatenate(
    [prices.reshape(-1, 1), smas, bollinger_bands_input, macds_input.reshape(-1, 1), rsi.reshape(-1, 1)], axis=1
)

normed_array, label_array = build_dataset_with_normalization_in_batches(
    input_array, k=360, p=120, batch_size=10000
)
# %%
model = build_hybrid_technical_model(seq_len=k)
# %%
train_ratio = 0.6
valid_ratio = 0.2
train_size = int(len(normed_array) * train_ratio)
valid_size = int(len(normed_array) * valid_ratio)
test_size = len(normed_array) - train_size - valid_size
train_x = normed_array[:train_size]
train_y = label_array[:train_size]
valid_x = normed_array[train_size:train_size + valid_size]
valid_y = label_array[train_size:train_size + valid_size]
test_x = normed_array[train_size + valid_size:]
test_y = label_array[train_size + valid_size:]
# %%
train_x.shape
# %%
print("[INFO] Starting training process...")

learning_rate_initial = 1e-3
learning_rate_final = 1e-5
switch_epoch = 200

trainer = Trainer(
    model=model,
    train_data=(train_x, train_y),
    valid_data=(valid_x, valid_y),
    test_data=(test_x, test_y),
    learning_rate_initial=learning_rate_initial,
    learning_rate_final=learning_rate_final,
    switch_epoch=switch_epoch,         # 学習率を切り替えるステップ数
    random_init_ratio=1e-4,   # バリデーション損失が改善しなくなった場合の部分的ランダム初期化率
    max_epochs=10000,
    patience=10,              # validationが改善しなくなってから再初期化までの猶予回数
    num_repeats=3,            # 学習→バリデーション→（初期化）を繰り返す試行回数
    batch_size=2000,
    early_stop_patience=25
)

trainer.run()
print("[INFO] Training finished.")

# %%
