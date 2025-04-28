# modules/models.py
"""モデル定義モジュール

   * ニューラルネットワークの構造を定義する関数をまとめる。
   * 例: 全結合モデル, Transformerなど
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np


class TechnicalIndicatorLayer(tf.keras.layers.Layer):
    """
    Computes four channels of technical indicators from an input price sequence:
      1) Original price
      2) SMA(window_sma)
      3) RSI(window_rsi)
      4) Bollinger band width(window_boll) = (UpperBand - LowerBand), where bands are mean ± 2*std

    Input shape:
        (batch_size, seq_length)

    Output shape:
        (batch_size, seq_length, 4)
    """

    def __init__(self, window_sma=20, window_rsi=14, window_boll=20):
        super().__init__()
        self.window_sma = window_sma
        self.window_rsi = window_rsi
        self.window_boll = window_boll

    def build(self, input_shape):
        # 1) SMA kernel
        self.sma_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.window_sma,
            padding='same',
            use_bias=False,
            trainable=False
        )
        w_sma = np.ones((self.window_sma, 1, 1),
                        dtype=np.float32) / self.window_sma
        self.sma_conv.build((None, None, 1))
        self.sma_conv.set_weights([w_sma])

        # 2) Bollinger mean kernel
        self.boll_mean_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.window_boll,
            padding='same',
            use_bias=False,
            trainable=False
        )
        w_boll_mean = np.ones((self.window_boll, 1, 1),
                              dtype=np.float32) / self.window_boll
        self.boll_mean_conv.build((None, None, 1))
        self.boll_mean_conv.set_weights([w_boll_mean])

        # 3) Bollinger variance kernel
        self.boll_var_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.window_boll,
            padding='same',
            use_bias=False,
            trainable=False
        )
        self.boll_var_conv.build((None, None, 1))
        self.boll_var_conv.set_weights([w_boll_mean])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.expand_dims(inputs, axis=-1)  # (batch, seq, 1)
        price = x

        sma = self.sma_conv(price)

        delta = price[:, 1:, :] - price[:, :-1, :]
        delta = tf.pad(delta, paddings=[[0, 0], [1, 0], [0, 0]])

        gains = tf.clip_by_value(delta, 0.0, np.inf)
        losses = tf.clip_by_value(-delta, 0.0, np.inf)

        gains_avg = tf.nn.avg_pool1d(
            gains,
            ksize=self.window_rsi,
            strides=1,
            padding='SAME'
        )
        losses_avg = tf.nn.avg_pool1d(
            losses,
            ksize=self.window_rsi,
            strides=1,
            padding='SAME'
        )

        eps = tf.constant(1e-8, dtype=gains_avg.dtype)
        rs = gains_avg / (losses_avg + eps)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        mean_boll = self.boll_mean_conv(price)
        squares = price * price
        mean_sq_boll = self.boll_var_conv(squares)
        var_boll = mean_sq_boll - (mean_boll * mean_boll)
        std_boll = tf.sqrt(tf.maximum(var_boll, 0.0))
        boll_width = 4.0 * std_boll

        out = tf.concat([price, sma, rsi, boll_width], axis=-1)
        return out


class SimpleAttention(tf.keras.layers.Layer):
    """
    A simple attention mechanism that merges two (batch, hidden_dim) vectors
    by learning separate trainable vectors to weight each branch output,
    then summing them.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        # input_shape is a list: [ (None, hidden_dim), (None, hidden_dim) ]
        # We'll create a trainable weight vector for each input.
        self.att_weights = []
        for i, shape in enumerate(input_shape):
            # shape[-1] = hidden_dim
            w = self.add_weight(
                shape=(shape[-1],),
                initializer="zeros",
                trainable=True,
                name=f"att_weight_{i}"
            )
            self.att_weights.append(w)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: [ x_lstm, x_cnn ]
        # each shape: (batch, hidden_dim)
        weighted = []
        for x, w in zip(inputs, self.att_weights):
            # x shape: (batch, hidden_dim)
            # w shape: (hidden_dim,)
            # multiply elementwise
            w_expanded = tf.expand_dims(w, axis=0)  # (1, hidden_dim)
            out = x * w_expanded
            weighted.append(out)

        # sum them
        return tf.add_n(weighted)


def build_lstm_cnn_attention_indicator_model(input_dim, window_sma=20, window_rsi=14, window_boll=20):
    """
    Builds a parallel LSTM+CNN model that internally computes the following technical indicators:
      - SMA(5)
      - RSI(14)
      - Bollinger band width(20)
    The final input shape to the parallel branches = (batch, seq_length, 4).
    The parallel outputs are merged via a simple attention mechanism,
    then a 2-class softmax is output.

    Args:
        input_dim (int): sequence length (k). Input shape => (batch, k)

    Returns:
        tf.keras.Model: A compiled Keras model with 2-class softmax output.
    """

    # 1) Input layer
    input_layer = tf.keras.Input(shape=(input_dim,), name="price_input")

    # 2) Technical indicators
    indicator_block = TechnicalIndicatorLayer(
        window_sma=window_sma, window_rsi=window_rsi, window_boll=window_boll)
    x_indicators = indicator_block(input_layer)  # shape (batch, seq, 4)

    # 3) LSTM branch
    x_lstm = tf.keras.layers.LSTM(64, return_sequences=False)(
        x_indicators)  # (batch, 64)

    # 4) CNN branch
    x_cnn = tf.keras.layers.Conv1D(
        filters=32, kernel_size=3, padding="same", activation="relu")(x_indicators)
    x_cnn = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same", activation="relu")(x_cnn)
    x_cnn = tf.keras.layers.GlobalAveragePooling1D()(x_cnn)  # (batch, 64)

    # 5) Attention
    att_layer = SimpleAttention(hidden_dim=64)
    x_att = att_layer([x_lstm, x_cnn])  # (batch, 64)

    # 6) Final dense
    x_dense = tf.keras.layers.Dense(32, activation="relu")(x_att)
    output_layer = tf.keras.layers.Dense(2, activation="softmax")(x_dense)

    # 7) Model compile
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer,
                           name="Parallel_LSTM_CNN_Attention_Model")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def build_hybrid_technical_model_(seq_len) -> tf.keras.Model:
    """
    入力 shape: (None, seq_len, 12)  - 正規化済みデータ
      0   : price
      1-6 : SMA diff 6ch
      7-9 : Bollinger diff 3ch
     10   : MACD diff
     11   : RSI (-1〜+1)
    出力: 上昇 / 下降 2クラス Softmax
    """
    inp = Input(shape=(seq_len, 12), name="input")

    # ------------- スライス -----------------
    price = layers.Lambda(lambda x: x[...,  0: 1])(inp)  # (B, L, 1)
    sma = layers.Lambda(lambda x: x[...,  1: 7])(inp)  # (B, L, 6)
    boll = layers.Lambda(lambda x: x[...,  7:10])(inp)  # (B, L, 3)
    macd = layers.Lambda(lambda x: x[..., 10:11])(inp)  # (B, L, 1)
    rsi = layers.Lambda(lambda x: x[..., 11:12])(inp)  # (B, L, 1)

    # ------------- Price branch (Conv→GRU) -------------
    x_p = layers.Conv1D(16, 10, padding="same", activation="relu")(price)
    x_p = layers.Conv1D(8, 5, padding="same", activation="relu")(x_p)
    x_p = layers.GRU(32, return_sequences=False)(x_p)
    x_p = layers.Dense(32, activation="relu")(x_p)

    # ------------- SMA branch (Conv×2→GAP) -------------
    x_s = layers.Conv1D(32, 10, padding="same", activation="relu")(sma)
    x_s = layers.Conv1D(16, 5, padding="same", activation="relu")(x_s)
    x_s = layers.GlobalAveragePooling1D()(x_s)
    x_s = layers.Dense(32, activation="relu")(x_s)

    # ------------- Boll branch (軽量Conv→GAP) ----------
    x_b = layers.Conv1D(16, 5, padding="same", activation="relu")(boll)
    x_b = layers.GlobalAveragePooling1D()(x_b)
    x_b = layers.Dense(32, activation="relu")(x_b)

    # ------------- MACD branch (GRU) -------------------
    x_m = layers.GRU(16, return_sequences=False)(macd)
    x_m = layers.Dense(32, activation="relu")(x_m)

    # ------------- RSI branch (Conv→GAP) ---------------
    x_r = layers.Conv1D(16, 5, padding="same", activation="relu")(rsi)
    x_r = layers.GlobalAveragePooling1D()(x_r)
    x_r = layers.Dense(32, activation="relu")(x_r)

    # ------------- 統合 (Add or Concat) ----------------
    fused = layers.Add()([x_p, x_s, x_b, x_m, x_r])

    # ------------- 全結合ヘッド ------------------------
    x = layers.BatchNormalization()(fused)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)

    out = layers.Dense(2, activation="softmax", name="direction")(x)

    model = Model(inputs=inp, outputs=out, name="Hybrid_TISplit_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_hybrid_technical_model(seq_len) -> tf.keras.Model:
    """
    入力 shape: (None, seq_len, 12)  - 正規化済みデータ
      0   : price
      1-6 : SMA diff 6ch
      7-9 : Bollinger diff 3ch
     10   : MACD diff
     11   : RSI (-1〜+1)
    出力: 上昇 / 下降 2クラス Softmax
    """
    inp = Input(shape=(seq_len, 12), name="input")

    # ------------- スライス -----------------
    price = layers.Lambda(lambda x: x[...,  0: 1])(inp)  # (B, L, 1)
    sma = layers.Lambda(lambda x: x[...,  1: 7])(inp)  # (B, L, 6)
    boll = layers.Lambda(lambda x: x[...,  7:10])(inp)  # (B, L, 3)
    macd = layers.Lambda(lambda x: x[..., 10:11])(inp)  # (B, L, 1)
    rsi = layers.Lambda(lambda x: x[..., 11:12])(inp)  # (B, L, 1)

    # ------------- Price branch (Conv→GRU) -------------
    x_p = layers.Conv1D(16, 10, padding="same", activation="relu")(price)
    x_p = layers.Conv1D(8, 5, padding="same", activation="relu")(x_p)
    x_p = layers.GRU(32, return_sequences=False)(x_p)
    x_p = layers.Dense(32, activation="relu")(x_p)

    # ------------- SMA branch (Conv×2→GAP) -------------
    x_s = layers.Conv1D(32, 10, padding="same", activation="relu")(sma)
    x_s = layers.Conv1D(16, 5, padding="same", activation="relu")(x_s)
    x_s = layers.GlobalAveragePooling1D()(x_s)
    x_s = layers.Dense(32, activation="relu")(x_s)

    # ------------- Boll branch (軽量Conv→GAP) ----------
    x_b = layers.Conv1D(16, 5, padding="same", activation="relu")(boll)
    x_b = layers.GlobalAveragePooling1D()(x_b)
    x_b = layers.Dense(32, activation="relu")(x_b)

    # ------------- MACD branch (GRU) -------------------
    x_m = layers.GRU(16, return_sequences=False)(macd)
    x_m = layers.Dense(32, activation="relu")(x_m)

    # ------------- RSI branch (Conv→GAP) ---------------
    x_r = layers.Conv1D(16, 5, padding="same", activation="relu")(rsi)
    x_r = layers.GlobalAveragePooling1D()(x_r)
    x_r = layers.Dense(32, activation="relu")(x_r)

    # ------------- 統合 (Add or Concat) ----------------
    fused = layers.Add()([x_p, x_s, x_b, x_m, x_r])

    # ------------- 全結合ヘッド ------------------------
    x = layers.BatchNormalization()(fused)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)

    out = layers.Dense(2, activation="softmax", name="direction")(x)

    model = Model(inputs=inp, outputs=out, name="Hybrid_TISplit_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
