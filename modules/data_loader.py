# modules/data_loader.py
"""データローダーモジュール

   * CSVファイルからtimestampとpriceを取得する機能。
"""

import pandas as pd


def load_csv_data(csv_path, skip=1):
    """CSVファイルを読み込み、timestampとpriceの配列を返す

    CSV形式:
        timestamp,price
        1579068180,1.11508
        1579068240,1.11509
        ...

    Args:
        csv_path (str): CSVファイルのパス

    Returns:
        (list of int, list of float):
            timestamps: 各行のタイムスタンプ
            prices: 各行の価格
    """
    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].tolist()
    prices = df["price"].tolist()
    return timestamps[::skip], prices[::skip]