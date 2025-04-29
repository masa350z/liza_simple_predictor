# %%
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from technical_funcs import calc_sma, calc_bollinger_bands, calc_macd, calc_rsi
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from datetime import datetime
from modules.data_loader import load_csv_data
from modules.models import build_hybrid_technical_model
import matplotlib.pyplot as plt

# 必要な分だけGPUメモリを確保する
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def make_hybrid_technical_model_data(prices,
                                     k, p,
                                     sma_short_window,
                                     sma_mid_window,
                                     sma_long_window,
                                     bollinger_band_window,
                                     bollinger_band_sigma,
                                     macd_short_window,
                                     macd_long_window,
                                     macd_signal_window,
                                     rsi_window):

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
        input_array, k=k, p=p, batch_size=10000
    )

    return normed_array, label_array
    # return input_array, _


def build_dataset_with_normalization_in_batches(input_array,
                                                k,
                                                p,
                                                batch_size=10000,
                                                rsi_idx=11,
                                                eps=1e-8):
    """
    ミニバッチで (N, k, 12) の正規化済みデータとラベルを生成(出力は float16)

    input_array: (T, 12) 価格 + テクニカル指標(RSIはインデックス11に想定)
    k: 入力系列長
    p: future予測系列差分
    batch_size: スライス単位の処理件数
    rsi_idx: RSIのチャネル位置(デフォルト: 11)
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


def split_data(data, train_ratio=0.6, valid_ratio=0.2):
    """
    データをトレーニング、バリデーション、テストセットに分割する関数
    """
    train_size = int(len(data) * train_ratio)
    valid_size = int(len(data) * valid_ratio)
    test_size = len(data) - train_size - valid_size

    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]

    return train_data, valid_data, test_data


def predict_in_batches(model, input_data, batch_size=1024):
    """モデルでミニバッチ推論を行い、結果を結合してNumPy配列として返す。

    Args:
        model (tf.keras.Model): 推論に使用するモデル
        input_data (np.ndarray or tf.Tensor): 入力データ (shape: [N, ...])
        batch_size (int, optional): ミニバッチサイズ (default=1024)

    Returns:
        np.ndarray: 結合された全推論結果 (shape: [N, クラス数])
    """
    total_size = len(input_data)
    predictions = []

    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batch = input_data[start:end]
        preds = model(batch, training=False)  # training=False で推論モード
        predictions.append(preds.numpy())     # Tensor → NumPyに変換して蓄積

    return np.concatenate(predictions, axis=0)


def print_accuracy_ratio(pred, label,
                         up_thresh_hold=0.5,
                         down_thresh_hold=0.5):
    up_ = label[:, 0]
    down_ = label[:, 1]

    long_position = 1*(pred[:, 0] > up_thresh_hold)
    short_position = 1*(pred[:, 1] > down_thresh_hold)

    long_position_count = np.sum(long_position)
    short_position_count = np.sum(short_position)

    accuracy_ratio_up = np.sum(long_position*up_)/np.sum(long_position)
    accuracy_ratio_down = np.sum(short_position*down_)/np.sum(short_position)

    print(f'long_position_count: {long_position_count}')
    print(f'short_position_count: {short_position_count}')

    print(f'accuracy_ratio_up: {accuracy_ratio_up}')
    print(f'accuracy_ratio_down: {accuracy_ratio_down}')


def print_all_results(data_x, model, train_ratio=0.6, valid_ratio=0.2):
    train_x, valid_x, test_x = split_data(
        data_x, train_ratio=train_ratio, valid_ratio=valid_ratio)
    train_y, valid_y, test_y = split_data(
        data_y, train_ratio=train_ratio, valid_ratio=valid_ratio)

    pred_train = predict_in_batches(model, train_x, batch_size=8000)
    pred_valid = predict_in_batches(model, valid_x, batch_size=8000)
    pred_test = predict_in_batches(model, test_x, batch_size=8000)

    print_accuracy_ratio(pred_train, train_y)
    print('==========================')
    print_accuracy_ratio(pred_valid, valid_y)
    print('==========================')
    print_accuracy_ratio(pred_test, test_y)
    print('==========================')


def run_simulation(row_prices, pred_all, max_hold_count,
                   rik_up, son_up, rik_dn, son_dn,
                   up_thresh_hold=0.5, down_thresh_hold=0.5, spread=0.03/100):
    kane = 0
    asset = []
    position = 0
    previous_position = 0
    get_price = 0
    count = 0

    for i in tqdm(range(len(pred_all))):
        if count > 0:
            count -= 1

            if count == 0:
                kane += (row_prices[i] - get_price)*position/get_price
                position = 0
            else:
                if position == 1:
                    if (row_prices[i]/get_price - 1 > rik_up) or (row_prices[i]/get_price - 1 < -son_up):
                        position = 0
                        kane += (row_prices[i] - get_price)/get_price

                elif position == -1:
                    if (get_price/row_prices[i] - 1 > rik_dn) or (get_price/row_prices[i] - 1 < -son_dn):
                        position = 0
                        kane += (get_price - row_prices[i])/get_price

        if position == 0:
            if pred_all[i, 0] > up_thresh_hold:
                position = 1
                get_price = row_prices[i]
                count = max_hold_count
            elif pred_all[i, 1] > down_thresh_hold:
                position = -1
                get_price = row_prices[i]
                count = max_hold_count

        if position != previous_position:
            kane -= row_prices[i]*spread/get_price
        previous_position = position

        asset.append(kane)

    asset = np.array(asset)

    return kane, asset


def save_simulation_result(pair: str,
                           model_class_name: str,
                           model_name: str,
                           m: int, k: int, p: int,
                           rik_up: float, son_up: float,
                           rik_dn: float, son_dn: float,
                           up_thresh_hold: float, down_thresh_hold: float,
                           final_kane: float,
                           asset_array: np.ndarray):
    """シミュレーション結果の保存(CSV+グラフ画像)

    Args:
        pair (str): 通貨ペア(例: "BTCJPY")
        model_class_name (str): モデルクラス名(例: "HybridTechnicalModel")
        model_path_name (str): モデルパス名
        m (int): ダウンサンプリング間隔
        k (int): 入力系列長
        p (int): 未来予測系列長
        rik_up, son_up, rik_dn, son_dn (float): 利確・損切り閾値
        up_thresh_hold, down_thresh_hold (float): 上昇/下降の判定閾値
        final_kane (float): 最終損益
        asset_array (np.ndarray): 時系列の資産推移(float可)
    """
    output_dir = os.path.join(
        "results", pair, model_class_name, "simulation_results")
    os.makedirs(output_dir, exist_ok=True)

    # === 資産推移保存 (float16 CSV) ===
    asset_csv_path = os.path.join(output_dir, f"{model_name}_asset_curve.csv")
    df_asset = pd.DataFrame(asset_array.astype(np.float16), columns=["asset"])
    df_asset.to_csv(asset_csv_path, index=False)

    # === 資産推移グラフ保存 (PNG) ===
    asset_png_path = os.path.join(output_dir, f"{model_name}_asset_curve.png")
    plt.figure(figsize=(12, 9))
    plt.plot(asset_array, label="Asset Curve")
    plt.title("Asset Over Time")
    plt.xlabel("Time")
    plt.ylabel("Asset")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(asset_png_path)
    plt.close()

    # === 結果要約をCSVに追記 ===
    summary_csv_path = os.path.join(
        "results", pair, f"simulation_results_{model_class_name}.csv")
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    file_exists = os.path.isfile(summary_csv_path)

    csv_row = {
        'Model_Name': output_dir,
        'm': m,
        'k': k,
        'future_k': p,
        'Rik_Up': rik_up,
        'Son_Up': son_up,
        'Rik_Dn': rik_dn,
        'Son_Dn': son_dn,
        'Up_Thresh': up_thresh_hold,
        'Down_Thresh': down_thresh_hold,
        'Final_Kane': f"{final_kane:.6f}",
    }

    with open(summary_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(csv_row.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)

    print(f"[INFO] Simulation result saved to: {output_dir}")
    print(f"[INFO] Asset curve saved: {asset_csv_path}")
    print(f"[INFO] Asset plot saved: {asset_png_path}")
    print(f"[INFO] Summary appended to: {summary_csv_path}")


# %%
pair = "BTCJPY"
csv_file_name = f"sample_{pair}_1m.csv"
csv_file_path = os.path.join("data", csv_file_name)

m = 2
k = 360
p = 120

model = build_hybrid_technical_model(seq_len=k)

model_path_base = 'results/BTCJPY/HybridTechnicalModel/weights/'
# model_name = 'm1_k360_f120_20250428-103133'
# model_name = 'm1_k360_f120_20250428-102353'
# model_name = 'm1_k360_f240_20250429-114844'
model_name = 'm2_k360_f120_20250429-122028'
model_path_name = model_path_base + model_name
model.load_weights(model_path_name + '/best_model.weights.h5')

_, prices_ = load_csv_data(csv_file_path, skip=1)
prices = np.array(prices_, dtype=np.float32)[::m]
data_x, data_y = make_hybrid_technical_model_data(prices,
                                                  k, p,
                                                  sma_short_window=20,
                                                  sma_mid_window=60,
                                                  sma_long_window=120,
                                                  bollinger_band_window=20,
                                                  bollinger_band_sigma=2.0,
                                                  macd_short_window=12,
                                                  macd_long_window=26,
                                                  macd_signal_window=9,
                                                  rsi_window=14
                                                  )
row_prices = prices[k-1:-p-1]

# print_all_results(data_x, model)

pred_all = predict_in_batches(model, data_x, batch_size=8000)
# %%
max_hold_count = 240

rik_up, son_up = 10/100, 5/100
rik_dn, son_dn = 6/100, 15/100

up_thresh_hold = 0.505
down_thresh_hold = 0.5


kane, asset = run_simulation(row_prices, pred_all, max_hold_count,
                             rik_up, son_up, rik_dn, son_dn,
                             up_thresh_hold, down_thresh_hold, spread=0.03/100)


pd.DataFrame(asset).plot()
print(kane)
# %%

# %%
save_simulation_result(pair=pair,
                       model_class_name="HybridTechnicalModel",
                       model_name=model_name,
                       m=m, k=k, p=p,
                       rik_up=rik_up, son_up=son_up,
                       rik_dn=rik_dn, son_dn=son_dn,
                       up_thresh_hold=up_thresh_hold,
                       down_thresh_hold=down_thresh_hold,
                       final_kane=kane,
                       asset_array=asset)

# %%
