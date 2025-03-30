# main.py
"""liza_trainer のエントリーポイント
学習ロジックを実行し、最終的にベストモデルを保存する。
"""
import os
import datetime
import tensorflow as tf
import csv

from modules.trainer import Trainer
from modules.models import build_simple_affine_model, build_lstm_cnn_attention_model, build_lstm_cnn_attention_indicator_model, build_transformer_ti_model
from modules.dataset import create_dataset
from modules.data_loader import load_csv_data


# 必要な分だけGPUメモリを確保する
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main(pair, m, k, future_k):
    """学習のメインフローを実行する関数。

    1. CSV読み込み (USDJPY or EURUSD)
    2. 特徴量/ラベル作成
    3. モデル定義
    4. 学習・評価のループ
    5. 保存用フォルダ作成 -> 最良モデル重み & infoを出力
    """

    # === 1. 通貨ペアの指定 (USDJPY or EURUSD) ===
    csv_file_name = f"sample_{pair}_1m.csv"
    csv_file = os.path.join("data", csv_file_name)
    print(f"[INFO] Loading CSV data for {pair} from: {csv_file}")

    timestamps, prices = load_csv_data(csv_file, skip=m)

    # === 2. 特徴量/ラベル作成 ===
    #   例: 直近k個の価格から future_k後の価格が上がるか(分類)
    print(f"[INFO] Creating dataset with k={k}, future_k={future_k}")
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = create_dataset(
        prices, k, future_k,
        train_ratio=0.6, valid_ratio=0.2, down_sampling=10
    )

    # === 3. モデル定義 ===
    model_class_name = "LSTM_CNN_INDICATOR"
    # model_class_name = "TRANSFORMER"
    print(f"[INFO] Building model ({model_class_name})")
    input_dim = train_x.shape[1]
    # model = build_simple_affine_model(input_dim)
    # model = build_lstm_cnn_attention_model(input_dim)
    model = build_lstm_cnn_attention_indicator_model(input_dim)
    # model = build_transformer_ti_model(input_dim)

    # === 4. 学習・評価のループ ===
    print("[INFO] Starting training process...")

    trainer = Trainer(
        model=model,
        train_data=(train_x, train_y),
        valid_data=(valid_x, valid_y),
        test_data=(test_x, test_y),
        learning_rate_initial=1e-3,
        learning_rate_final=1e-4,
        switch_epoch=100,         # 学習率を切り替えるステップ数
        random_init_ratio=1e-4,   # バリデーション損失が改善しなくなった場合の部分的ランダム初期化率
        max_epochs=10000,
        patience=10,              # validationが改善しなくなってから再初期化までの猶予回数
        num_repeats=5,            # 学習→バリデーション→（初期化）を繰り返す試行回数
        batch_size=4000,
        early_stop_patience=25
    )

    trainer.run()
    print("[INFO] Training finished.")

    # === 5. 保存用フォルダ作成 & 結果出力 ===
    # サブディレクトリ: "results/{pair}/{ModelClass}_{YYYYMMDD-HHMMSS}/"
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # ディレクトリ名に k, future_k の値を埋め込む
    model_name = f"m{m}_k{k}_f{future_k}_{now_str}"
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
    csv_path = os.path.join("results", pair, "training_results.csv")

    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # ファイルが存在するかチェック
    file_exists = os.path.isfile(csv_path)

    # CSVに記録する情報を整理
    csv_row = {
        'Pair': pair,
        'Model_Name': output_dir,
        'Best_Validation_Loss': f"{trainer.best_val_loss:.6f}",
        'Best_Validation_Acc': f"{trainer.best_val_acc:.6f}",
        'Best_Test_Loss': f"{trainer.best_test_loss:.6f}",
        'Best_Test_Acc': f"{trainer.best_test_acc:.6f}",
        'm': m,
        'k': k,
        'future_k': future_k
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


if __name__ == "__main__":
    m = 100
    for pair in ['BTCJPY']:
        for k in [360, 300, 240, 180, 120]:
            for i in [1, 2, 3]:
                future_k = int(k/i)

                main(pair, m, k, future_k)
