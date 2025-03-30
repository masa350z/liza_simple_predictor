# summary.py
"""
EURUSD / USDJPY の学習ログ (training_info.txt) をそれぞれ別のCSVに出力するスクリプト。
ディレクトリ名 (例: Affine_k30_f10_20250126-230851) からも k, future_k を補完する。
"""

import os
import re
import glob
import csv

def parse_training_info(txt_path):
    """
    training_info.txt を読み込み、必要情報を辞書にして返す。
    期待フォーマット例:
      Pair: EURUSD
      Model Class: Affine
      DateTime: 20250101-100000

      Hyperparameters:
        k = 30
        future_k = 5

      Best Validation Loss : 0.123456
      Best Validation Acc  : 0.789012
      Best Test Loss       : 0.234567
      Best Test Acc        : 0.678901

    さらに、ディレクトリ名 (例: Affine_k30_f10_20250126-230851) に
    "k\d+" や "f\d+" があればそれもパースして k / future_k に補完する。
    """
    info = {
        'pair': None,
        'model_class': None,
        'datetime': None,
        'k': None,
        'future_k': None,
        'best_val_loss': None,
        'best_val_acc': None,
        'best_test_loss': None,
        'best_test_acc': None
    }

    # 1) training_info.txt の内容を読む
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line.startswith("Pair:"):
            info['pair'] = line.split(":", 1)[1].strip()
        elif line.startswith("Model Class:"):
            info['model_class'] = line.split(":", 1)[1].strip()
        elif line.startswith("DateTime:"):
            info['datetime'] = line.split(":", 1)[1].strip()

        elif re.search(r"^\s*k\s*=\s*\d+", line):
            parts = line.split("=")
            if len(parts) == 2:
                info['k'] = int(parts[1].strip())
        elif re.search(r"^\s*future_k\s*=\s*\d+", line):
            parts = line.split("=")
            if len(parts) == 2:
                info['future_k'] = int(parts[1].strip())

        elif line.startswith("Best Validation Loss"):
            val_str = line.split(":", 1)[1].strip()
            info['best_val_loss'] = float(val_str)
        elif line.startswith("Best Validation Acc"):
            val_str = line.split(":", 1)[1].strip()
            info['best_val_acc'] = float(val_str)
        elif line.startswith("Best Test Loss"):
            val_str = line.split(":", 1)[1].strip()
            info['best_test_loss'] = float(val_str)
        elif line.startswith("Best Test Acc"):
            val_str = line.split(":", 1)[1].strip()
            info['best_test_acc'] = float(val_str)

    # 2) ディレクトリ名から k, future_k を補完
    #    例: "Affine_k30_f10_20250126-230851" というフォルダ名
    #    base_name = "Affine_k30_f10_20250126-230851"
    dir_name = os.path.basename(os.path.dirname(txt_path))  # 親フォルダ名

    # k(\d+) と f(\d+) を拾うための正規表現
    #   _k30_  または  _k30_f10_ などを抜き出す
    #   _f(\d+) とか
    # 例:  "Affine_k30_f10_20250126-230851"
    #      => k=30, future_k=10
    k_match = re.search(r"_k(\d+)", dir_name)
    f_match = re.search(r"_f(\d+)", dir_name)

    if info['k'] is None and k_match:
        info['k'] = int(k_match.group(1))  # "k30" => 30
    if info['future_k'] is None and f_match:
        info['future_k'] = int(f_match.group(1))

    return info

def gather_info_for_pair(pair):
    base_dir = "results"
    pair_dir = os.path.join(base_dir, pair)
    if not os.path.isdir(pair_dir):
        return []

    pattern = os.path.join(pair_dir, "*", "training_info.txt")
    txt_paths = glob.glob(pattern)

    info_list = []
    for txt_path in txt_paths:
        info = parse_training_info(txt_path)
        if info['pair'] is None:  # フォーマット不備など
            continue
        info_list.append(info)

    # ソート例: model_class, datetime (などお好みで)
    info_list.sort(key=lambda x: (x['model_class'] or "", x['datetime'] or ""))
    return info_list

def write_csv_for_pair(pair, info_list):
    out_file = f"results/{pair}/summary_{pair}.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow([
            "pair", "model_class", "datetime", "k", "future_k",
            "best_val_loss", "best_val_acc", "best_test_loss", "best_test_acc"
        ])
        for info in info_list:
            writer.writerow([
                info['pair'],
                info['model_class'],
                info['datetime'],
                info['k'],
                info['future_k'],
                info['best_val_loss'],
                info['best_val_acc'],
                info['best_test_loss'],
                info['best_test_acc']
            ])

    print(f"[INFO] {pair} -> wrote {len(info_list)} records to {out_file}")

def main():
    pairs = ["EURUSD", "USDJPY", "BTCJPY"]
    for pair in pairs:
        info_list = gather_info_for_pair(pair)
        if len(info_list) == 0:
            print(f"[WARN] No logs found for {pair}")
            continue
        write_csv_for_pair(pair, info_list)

if __name__ == "__main__":
    main()
