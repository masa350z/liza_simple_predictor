from tensorflow.keras import layers, Model, Input
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from modules.data_loader import load_csv_data


def calc_sma(prices, window=20):
    """
    単純移動平均 (SMA) を計算し、出力の長さを `len(prices)` に揃える。
    最初の (window - 1) 要素は 0.0 で埋める。

    Args:
        prices (np.ndarray): shape = (N,)
        window (int): 移動平均の期間

    Returns:
        np.ndarray: shape = (N,), 価格と同じスケール
    """
    N = len(prices)
    sma = np.zeros(N, dtype=np.float64)  # 出力用
    if window <= 0 or window > N:
        return sma  # 不正な場合は全ゼロ

    # 累積和を使って高速に平均を計算
    cumsum = np.cumsum(prices, dtype=np.float64)
    # SMA計算 => [ (cumsum[window-1:] - cumsum[:-window]) / window ]
    for i in range(window - 1, N):
        if i == (window - 1):
            sma[i] = cumsum[i] / window
        else:
            sma[i] = (cumsum[i] - cumsum[i - window]) / window

    return sma.astype(np.float32)


def calc_bollinger_bands(prices, window=20, multi=2.0):
    """
    ボリンジャーバンドを計算し、(center, upper, lower) 3つを返す。
    各配列は shape=(N,), 先頭 (window-1) 要素は 0 埋め。
    center: SMA
    upper: SMA + multi * std
    lower: SMA - multi * std
    """
    N = len(prices)
    center = calc_sma(prices, window)  # 同じスケール
    upper = np.zeros(N, dtype=np.float64)
    lower = np.zeros(N, dtype=np.float64)

    if window <= 0 or window > N:
        return center, upper, lower

    # rolling std を計算し、先頭 (window-1) は 0 に
    rolling_std = np.zeros(N, dtype=np.float64)

    # 累積和の2乗を使って分散を高速計算
    cumsum = np.cumsum(prices, dtype=np.float64)
    cumsum_sq = np.cumsum(prices**2, dtype=np.float64)

    for i in range(window - 1, N):
        window_sum = cumsum[i] - (0 if i-window < 0 else cumsum[i-window])
        window_sq_sum = cumsum_sq[i] - \
            (0 if i-window < 0 else cumsum_sq[i-window])
        mean = window_sum / window
        # 分散 = (x^2の平均) - (平均)^2
        var = (window_sq_sum / window) - (mean**2)
        std_ = np.sqrt(var) if var > 0 else 0.0
        rolling_std[i] = std_

    # upper / lower
    upper = center + multi * rolling_std
    lower = center - multi * rolling_std

    return center.astype(np.float32), upper.astype(np.float32), lower.astype(np.float32)


def calc_macd(prices, short_window=12, long_window=26, signal_window=9):
    """
    MACDラインとシグナル線を計算し、それぞれを価格平均でオフセットして返す。
    (macd_line, signal_line) の2つを返却。
    先頭の (long_window - 1) 要素とシグナル計算分の要素は 0 埋め。

    Args:
        prices (np.ndarray): shape=(N,)
        short_window (int): 短期EMA期間
        long_window (int): 長期EMA期間
        signal_window (int): シグナル線のEMA期間

    Returns:
        (np.ndarray, np.ndarray):
            macd_line_scaled, signal_line_scaled
            ともに shape=(N,)
            先頭部は必要に応じて0埋め。
    """
    N = len(prices)
    if N < long_window:
        return np.zeros(N), np.zeros(N)

    # EMA計算関数
    def ema(arr, window):
        alpha = 2.0 / (window + 1.0)
        res = np.zeros_like(arr, dtype=np.float64)
        res[0] = arr[0]
        for i in range(1, len(arr)):
            res[i] = alpha * arr[i] + (1 - alpha) * res[i-1]
        return res

    # 短期EMA, 長期EMA
    short_ema = ema(prices, short_window)
    long_ema = ema(prices, long_window)
    macd_line = short_ema - long_ema  # shape=(N,)

    # シグナル線
    signal_line = ema(macd_line, signal_window)

    # 先頭に 0埋めをする分は => 実際には長EMAぶん or max(short_window, long_window)
    # ただし単純化のため、long_window - 1 個ぶんを 0埋め
    # シグナル線も signal_window を考慮して、先頭にさらに 0埋め
    macd_line_final = np.zeros(N, dtype=np.float64)
    signal_line_final = np.zeros(N, dtype=np.float64)

    # macdラインは (long_window-1) 以降を反映
    start_idx = long_window - 1
    macd_line_final[start_idx:] = macd_line[start_idx:]

    # シグナル線は (long_window -1 + signal_window -1) 以降を反映
    sig_start_idx = long_window - 1 + signal_window - 1
    signal_line_final[sig_start_idx:] = signal_line[sig_start_idx:]

    # 価格平均でオフセット
    p_mean = np.mean(prices)
    macd_line_scaled = macd_line_final + p_mean
    signal_line_scaled = signal_line_final + p_mean

    return macd_line_scaled.astype(np.float32), signal_line_scaled.astype(np.float32)


def calc_rsi(prices, window=14):
    """
    RSI(相対力指数)を計算し、配列長を len(prices) に揃えて先頭 (window-1) を0埋め。
    値域は [0, 100] (オシレーター型)

    Args:
        prices (np.ndarray): (N,)
        window (int): RSI期間 (一般的には14)

    Returns:
        np.ndarray: shape=(N,), 先頭は0埋め
    """
    N = len(prices)
    if N < window:
        return np.zeros(N, dtype=np.float64)

    rsi_arr = np.zeros(N, dtype=np.float64)
    # 差分
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    # 平均利得・損失（最初のwindowぶんだけSMAで計算）
    avg_gain = np.sum(gains[:window]) / window
    avg_loss = np.sum(losses[:window]) / window

    # 最初の (window) 時点のRSI
    rs = avg_gain / (avg_loss + 1e-8)
    rsi_arr[window] = 100.0 - (100.0 / (1.0 + rs))

    # 以降は徐々に更新
    for i in range(window+1, N):
        gain = gains[i-1]
        loss = losses[i-1]
        # 14日RSIの場合、「スムーズな平均」を計算
        avg_gain = (avg_gain*(window-1) + gain) / window
        avg_loss = (avg_loss*(window-1) + loss) / window
        rs = avg_gain / (avg_loss + 1e-8)
        rsi_arr[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi_arr.astype(np.float32)
