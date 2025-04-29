# %%
from modules.trainer import Trainer
from technical_funcs import calc_sma, calc_bollinger_bands, calc_macd, calc_rsi
import tensorflow as tf
import csv
import os
import numpy as np
from datetime import datetime
from modules.data_loader import load_csv_data
from modules.dataset import _balance_up_down
from modules.models import build_hybrid_technical_model

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


def save_csv(best_val_loss, best_val_acc, best_test_loss, best_test_acc):
    # === 5. 保存用フォルダ作成 & 結果出力 ===
    # サブディレクトリ: "results/{pair}/{ModelClass}_{YYYYMMDD-HHMMSS}/"
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ディレクトリ名に k, future_k の値を埋め込む
    model_name = f"m{m}_k{k}_f{p}_{now_str}"
    output_dir = os.path.join(
        "results",
        pair,
        model_class_name,
        model_name
    )
    os.makedirs(output_dir, exist_ok=True)

    # bestモデルの重み保存
    best_weights_path = os.path.join(output_dir, "best_model.weights.h5")
    trainer.save_best_weights(best_weights_path)

    # === 6. 結果をCSVにまとめて出力 ===
    csv_path = os.path.join(
        "results", pair, f"training_results_{model_class_name}.csv")

    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # ファイルが存在するかチェック
    file_exists = os.path.isfile(csv_path)

    # CSVに記録する情報を整理
    csv_row = {
        'Model_Name': output_dir,
        'm': m,
        'k': k,
        'future_k': p,
        'Best_Validation_Loss': f"{best_val_loss:.6f}",
        'Best_Validation_Acc': f"{best_val_acc:.6f}",
        'Best_Test_Loss': f"{best_test_loss:.6f}",
        'Best_Test_Acc': f"{best_test_acc:.6f}",
        'Early_Stop_Patience': trainer.early_stop_patience,
        'down_sampling': skip_num,
        'Switch_Eochs': switch_epoch,
        'learning_rate_initial': learning_rate_initial,
        'learning_rate_final': learning_rate_final,
        'SMA': f"{sma_short_window}, {sma_mid_window}, {sma_long_window}",
        'BOLL': f"{bollinger_band_window}, {bollinger_band_sigma}",
        'MACD': f"{macd_short_window}, {macd_long_window}, {macd_signal_window}",
        'RSI': f"{rsi_window}",
    }

    # CSVに書き込み（存在しない場合はヘッダーも書き込み）
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(csv_row.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # ファイルが存在しない場合はヘッダーを書き込む
        if not file_exists:
            writer.writeheader()

        writer.writerow(csv_row)

    print(f"[INFO] Summary results appended to: {csv_path}")


# %%
m = 2
pair = "BTCJPY"
csv_file_name = f"sample_{pair}_1m.csv"
csv_file_path = os.path.join("data", csv_file_name)

k = 360
p = 90

sma_short_window = 20
sma_mid_window = 60
sma_long_window = 120

bollinger_band_window = 20
bollinger_band_sigma = 2.0


macd_short_window = 12
macd_long_window = 26
macd_signal_window = 9

rsi_window = 14

model_class_name = "HybridTechnicalModel"
model = build_hybrid_technical_model(seq_len=k)

_, prices_ = load_csv_data(csv_file_path, skip=1)
prices = np.array(prices_, dtype=np.float32)[::m]
data_x, data_y = make_hybrid_technical_model_data(prices,
                                                  k, p,
                                                  sma_short_window,
                                                  sma_mid_window,
                                                  sma_long_window,
                                                  bollinger_band_window,
                                                  bollinger_band_sigma,
                                                  macd_short_window,
                                                  macd_long_window,
                                                  macd_signal_window,
                                                  rsi_window
                                                  )

train_ratio = 0.6
valid_ratio = 0.2

train_x, valid_x, test_x = split_data(
    data_x, train_ratio=train_ratio, valid_ratio=valid_ratio)
train_y, valid_y, test_y = split_data(
    data_y, train_ratio=train_ratio, valid_ratio=valid_ratio)

train_x, train_y = _balance_up_down(train_x, train_y)
valid_x, valid_y = _balance_up_down(valid_x, valid_y)
test_x, test_y = _balance_up_down(test_x, test_y)
# %%
skip_num = 20

train_x = train_x[::skip_num]
train_y = train_y[::skip_num]
valid_x = valid_x[::skip_num]
valid_y = valid_y[::skip_num]
test_x = test_x[::skip_num]
test_y = test_y[::skip_num]
# %%
print("[INFO] Starting training process...")

learning_rate_initial = 1e-3
learning_rate_final = 1e-5
switch_epoch = 200

while True:
    trainer = Trainer(
        model=model,
        train_data=(train_x, train_y),
        valid_data=(valid_x, valid_y),
        test_data=(test_x, test_y),
        learning_rate_initial=learning_rate_initial,
        learning_rate_final=learning_rate_final,
        switch_epoch=switch_epoch,         # 学習率を切り替えるステップ数
        max_epochs=10000,
        batch_size=4000,
        early_stop_patience=25
    )

    best_val_loss, best_val_acc, best_test_loss, best_test_acc = trainer.run()
    print("[INFO] Training finished.")

    save_csv(best_val_loss, best_val_acc, best_test_loss, best_test_acc)
