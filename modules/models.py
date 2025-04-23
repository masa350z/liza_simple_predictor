# modules/models.py
"""モデル定義モジュール

   * ニューラルネットワークの構造を定義する関数をまとめる。
   * 例: 全結合モデル, Transformerなど
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, Input
import numpy as np


def build_simple_affine_model(input_dim):
    """単純な全結合モデルを構築して返す

    Args:
        input_dim (int): 入力次元 (kに相当)

    Returns:
        tf.keras.Model: コンパイル済みのKerasモデル (2クラス分類)
    """
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax")
    ])
    return model


def build_simple_lstm_model(input_dim):
    """シンプルなLSTMモデルを構築して返す

    Args:
        input_dim (int): 過去の時系列長(k)に相当

    Returns:
        tf.keras.Model: コンパイル済みのKerasモデル (2クラス分類, softmax)
    """
    model = Sequential([
        # (batch, input_dim) を (batch, input_dim, 1) に変形
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),
        # LSTM部
        layers.LSTM(4, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax")
    ])
    return model


def build_simple_cnn_model(input_dim):
    """シンプルな1次元CNNモデルを構築して返す

    Args:
        input_dim (int): 過去の時系列長(k)に相当

    Returns:
        tf.keras.Model: コンパイル済みのKerasモデル (2クラス分類, softmax)
    """
    model = Sequential([
        # (batch, input_dim) を (batch, input_dim, 1) に変形
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),
        # Conv1D部
        layers.Conv1D(filters=32, kernel_size=3,
                      padding="causal", activation="relu"),
        layers.Conv1D(filters=64, kernel_size=3,
                      padding="causal", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax")
    ])
    return model


def build_lstm_cnn_attention_model(input_dim):
    """価格配列 (batch, k) に対し、NN内部で簡易的なテクニカル指標(SMAなど)を生成し、
    LSTM と CNN の2つのブランチを並列適用し、最終的にAttentionで統合するモデル。

    Args:
        input_dim (int): 過去k分(例: 90分) の価格を格納した入力の次元数

    Returns:
        tf.keras.Model: 2クラス分類(上昇/下降)のsoftmax出力を持つKerasモデル
    """

    class SimpleAttention(layers.Layer):
        """LSTMブランチとCNNブランチの出力ベクトル(同次元)を
        重み付けして足し合わせる簡単なAttentionレイヤー。"""

        def __init__(self, output_dim, **kwargs):
            super(SimpleAttention, self).__init__(**kwargs)
            self.output_dim = output_dim

        def build(self, input_shape):
            # input_shapeは [ (None, hidden_dim), (None, hidden_dim) ] のリスト想定
            # ブランチ数ぶんだけtrainableな重みベクトルを定義する
            self.attention_w = []
            for i, shape in enumerate(input_shape):
                # shape[-1] は各ブランチの出力次元
                w = self.add_weight(
                    name=f"att_weight_{i}",
                    shape=(shape[-1],),  # 単純化のため 各ブランチの出力と同じ次元のベクトル
                    initializer="zeros",
                    trainable=True
                )
                self.attention_w.append(w)

            super(SimpleAttention, self).build(input_shape)

        def call(self, inputs, **kwargs):
            # inputsは [x_lstm, x_cnn] など複数ブランチの出力テンソル
            # shape: (batch, hidden_dim) が2つ想定
            weighted_outs = []
            for x, w in zip(inputs, self.attention_w):
                # 内積をとって重み付け
                # wを(xと同形)にブロードキャストするため expand_dims などで次元合わせ
                weighted_outs.append(x * tf.expand_dims(w, axis=0))

            # 足し合わせて最終的な出力に
            merged = tf.add_n(weighted_outs)
            return merged

    # ----------------------------------------------------
    # 1) 入力: 形状 (batch, input_dim) を (batch, input_dim, 1) に変形
    # ----------------------------------------------------
    input_layer = layers.Input(shape=(input_dim,), name="price_input")
    # reshape -> (batch, k, 1)
    x_reshape = layers.Reshape((input_dim, 1))(input_layer)

    # ----------------------------------------------------
    # 2) NN内で簡易的なテクニカル指標(SMA)を固定Conv1Dで生成
    #    例: SMA(5), SMA(15), SMA(30) を計算 (パラメータは例)
    # ----------------------------------------------------
    def build_fixed_sma_conv(window_size):
        """window_sizeの単純移動平均をConv1Dで近似するレイヤーを作る。
           カーネルは1/window_size で固定し、学習しないようにする。
        """
        conv = layers.Conv1D(
            filters=1,
            kernel_size=window_size,
            padding='same',
            use_bias=False,
            trainable=False,
            name=f"fixed_sma_{window_size}"
        )
        # Conv1Dの重みを(カーネルサイズ, in_channels, out_channels)で上書き
        # カーネルサイズ=window_size, in_channels=1, out_channels=1
        # 単純平均なので 1/window_size を並べたものをセットする
        kernel_value = np.ones((window_size, 1, 1),
                               dtype=np.float32) / window_size
        conv.build((None, input_dim, 1))  # 重み初期化
        conv.set_weights([kernel_value])
        return conv

    sma_5 = build_fixed_sma_conv(5)(x_reshape)    # (batch, k, 1)
    sma_15 = build_fixed_sma_conv(15)(x_reshape)  # (batch, k, 1)
    sma_30 = build_fixed_sma_conv(30)(x_reshape)  # (batch, k, 1)

    # 元の価格チャネル x_reshape と SMAチャネル3つを結合 -> shape (batch, k, 4)
    x_concat = layers.Concatenate(axis=-1)([x_reshape, sma_5, sma_15, sma_30])

    # ----------------------------------------------------
    # 3) LSTMブランチ
    # ----------------------------------------------------
    x_lstm = layers.LSTM(64, return_sequences=False,
                         name="lstm_branch")(x_concat)
    # shape: (batch, 64)

    # ----------------------------------------------------
    # 4) CNNブランチ
    # ----------------------------------------------------
    x_cnn = layers.Conv1D(
        32, kernel_size=3, padding='causal', activation='relu')(x_concat)
    x_cnn = layers.Conv1D(
        64, kernel_size=3, padding='causal', activation='relu')(x_cnn)
    x_cnn = layers.GlobalAveragePooling1D()(x_cnn)
    # shape: (batch, 64)

    # ----------------------------------------------------
    # 5) Attentionレイヤーで統合
    # ----------------------------------------------------
    att_layer = SimpleAttention(output_dim=64)
    x_att = att_layer([x_lstm, x_cnn])  # shape: (batch, 64)

    # ----------------------------------------------------
    # 6) 出力層(2ユニットsoftmax)
    # ----------------------------------------------------
    x_dense = layers.Dense(32, activation="relu")(x_att)
    output_layer = layers.Dense(2, activation="softmax")(x_dense)

    # ----------------------------------------------------
    # 7) モデル定義 & コンパイル
    # ----------------------------------------------------
    model = Model(inputs=input_layer, outputs=output_layer,
                  name="LSTM_CNN_Attn_Model")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


##############################################################################
# Technical Indicator Layer
##############################################################################


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


##############################################################################
# Attention Layer
##############################################################################

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


##############################################################################
# Full Model: Parallel LSTM + CNN + Attention, with internal indicator generation
##############################################################################

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
        filters=32, kernel_size=3, padding="causal", activation="relu")(x_indicators)
    x_cnn = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="causal", activation="relu")(x_cnn)
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


class TechnicalIndicatorLayer02(tf.keras.layers.Layer):
    """
    Keras Layerでボリンジャーバンド(期間20, ±2σ幅) と ATR(期間14相当) を
    (batch, seq_length) → (batch, seq_length, 3) で計算する例。

    出力チャネル:
      0: 元の価格
      1: Bollinger幅(=4*std)
      2: ATRもどき(平均絶対差分14)

    内部では Conv1D(use_bias=False, trainable=False) を使ってrolling平均・分散を実装。
    'padding=same' により時系列長を変えずにゼロパディングを行います。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat3 = tf.keras.layers.Concatenate(axis=-1)  # 3チャネル用結合レイヤ

    def build(self, input_shape):
        # input_shape = (batch_size, seq_length)

        # 1) Conv1D kernel for rolling mean(20)
        self.boll_mean_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=20,
            padding='same',
            use_bias=False,
            trainable=False
        )
        w_boll_mean = np.ones((20, 1, 1), dtype=np.float32) / 20.0
        self.boll_mean_conv.build((None, None, 1))
        self.boll_mean_conv.set_weights([w_boll_mean])

        # 2) rolling mean of squares for std(20)
        self.boll_var_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=20,
            padding='same',
            use_bias=False,
            trainable=False
        )
        self.boll_var_conv.build((None, None, 1))
        self.boll_var_conv.set_weights([w_boll_mean])

        # 3) ATR(14) -like rolling average of |diff|
        self.atr_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=14,
            padding='same',
            use_bias=False,
            trainable=False
        )
        w_atr = np.ones((14, 1, 1), dtype=np.float32) / 14.0
        self.atr_conv.build((None, None, 1))
        self.atr_conv.set_weights([w_atr])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # (batch, seq_length) => (batch, seq_length, 1)
        price = tf.keras.layers.Reshape((-1, 1))(inputs)  # Keras層でのReshape

        # --- Bollinger(20) ---
        m20 = self.boll_mean_conv(price)  # rolling mean
        sq = price * price
        msq20 = self.boll_var_conv(sq)    # mean of squares
        var_20 = msq20 - (m20 * m20)
        std_20 = tf.sqrt(tf.maximum(var_20, 0.0))
        boll_width = 4.0 * std_20  # (batch, seq, 1)

        # --- ATR(14)-like ---
        # diff = | p_t - p_(t-1) |
        shifted = tf.keras.layers.Lambda(
            lambda x: tf.concat([x[:, :1, :], x[:, :-1, :]], axis=1)
        )(price)
        diff_ = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]))(([price, shifted]))
        atr_14 = self.atr_conv(diff_)  # (batch, seq, 1)

        # 最後に3チャネルをConcatenate
        out = self.concat3([price, boll_width, atr_14])  # (batch, seq, 3)
        return out


##############################################################################
# Time2Vec Layer
##############################################################################

class Time2Vec(tf.keras.layers.Layer):
    """
    シンプルなTime2Vec実装 (出力次元 out_dim)。
    入力 shape: (batch, seq, channels) を仮定。
    seq次元に対して [0..seq-1] をシンボリックに生成し、sin/cos風の変換を行う。
    """

    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        # input_shape = (batch, seq, c)
        self.w0 = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name="time2vec_w0"
        )
        self.b0 = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name="time2vec_b0"
        )
        self.w = self.add_weight(
            shape=(self.out_dim - 1,),
            initializer="glorot_uniform",
            trainable=True,
            name="time2vec_w"
        )
        self.b = self.add_weight(
            shape=(self.out_dim - 1,),
            initializer="zeros",
            trainable=True,
            name="time2vec_b"
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs.shape: (batch, seq, c)
        # seq長をsymbolicに取得
        seq_len = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0]

        # steps = [0,1,2,...,seq_len-1], shape=(seq_len,)
        rng = tf.range(0, seq_len, dtype=tf.float32)
        # reshape => (1, seq_len, 1), then tile to (batch, seq_len, 1)
        rng = tf.reshape(rng, (1, seq_len, 1))
        rng = tf.tile(rng, [batch_size, 1, 1])  # (batch, seq, 1)

        # linear part: w0 * t + b0
        linear_part = self.w0 * rng + self.b0  # (batch, seq, 1)

        # sin part: sin( w_k * t + b_k ) for k in [1..out_dim-1]
        w_ = tf.reshape(self.w, (1, 1, self.out_dim - 1))
        b_ = tf.reshape(self.b, (1, 1, self.out_dim - 1))
        # broadcast rng => (batch, seq, out_dim-1)
        rng_ = tf.tile(rng, [1, 1, self.out_dim - 1])
        # period part
        period_part = tf.sin(w_ * rng_ + b_)

        # concat => (batch, seq, out_dim)
        return tf.keras.layers.Concatenate(axis=-1)([linear_part, period_part])


##############################################################################
# Transformer Encoder
##############################################################################

class TransformerEncoder(tf.keras.layers.Layer):
    """
    1層のTransformerEncoder:
      - MultiHeadAttention (with residual + dropout + LayerNorm)
      - FeedForward (2層) (with residual + dropout + LayerNorm)
    """

    def __init__(self, d_model, num_heads, d_ff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=rate
        )
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        # x.shape: (batch, seq, d_model)
        attn_output = self.mha(x, x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.norm2(out1 + ffn_out)

        return out2


##############################################################################
# Full Model: Two-level Transformer with TI (BB+ATR) & Time2Vec
##############################################################################

def build_transformer_ti_model(input_dim):
    """
    入力: (batch, input_dim) の価格ベクトルを受け取り、
      1) BollingerBand+ATR (3チャネル)
      2) Time2Vec (5チャネル例)
    を合体して (batch, seq, 8) → Dense(16) → 
    Level1 TransformerEncoder * 3層 → 
    Level2 TransformerEncoder * 4層 →
    GlobalAveragePooling -> Dense -> 最終出力(1ユニット)
    """

    price_input = tf.keras.Input(shape=(input_dim,), name="price_input")

    # --- Technical Indicator Layer ---
    ti_layer = TechnicalIndicatorLayer02()
    x_ti = ti_layer(price_input)  # (batch, seq, 3)

    # --- Time2Vec(5次元) ---
    t2v = Time2Vec(5)
    x_time = t2v(x_ti)  # (batch, seq, 5)

    # merge => (batch, seq, 8) via Keras' Concatenate
    merged = tf.keras.layers.Concatenate(axis=-1)([x_ti, x_time])

    # project to d_model=16
    proj = tf.keras.layers.Dense(16)(merged)  # (batch, seq, 16)

    # Level1 TNN (3 encoder blocks)
    x = proj
    for i in range(3):
        enc = TransformerEncoder(d_model=16, num_heads=2, d_ff=32, rate=0.2)
        x = enc(x)

    # Level2 TNN (4 encoder blocks)
    for i in range(4):
        enc2 = TransformerEncoder(d_model=16, num_heads=2, d_ff=64, rate=0.2)
        x = enc2(x)

    # Pool over seq dimension
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # MLP -> single regression output
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=price_input, outputs=out, name="TNN_BB_ATR_model")

    return model


def build_hybrid_technical_model(seq_len) -> tf.keras.Model:
    """
    入力 shape: (None, seq_len, 12)  – 正規化済みデータ
      0   : price
      1-6 : SMA diff 6ch
      7-9 : Bollinger diff 3ch
     10   : MACD diff
     11   : RSI (−1〜+1)
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
    x_p = layers.Conv1D(32, 3, padding="causal", activation="relu")(price)
    x_p = layers.Conv1D(32, 3, padding="causal", dilation_rate=2,
                        activation="relu")(x_p)
    x_p = layers.GRU(64, return_sequences=False)(x_p)          # (B, 64)
    x_p = layers.Dense(32, activation="relu")(x_p)             # (B, 32)

    # ------------- SMA branch (Conv×2→GAP) -------------
    x_s = layers.Conv1D(32, 3, padding="causal", activation="relu")(sma)
    x_s = layers.Conv1D(32, 3, padding="causal", activation="relu")(x_s)
    x_s = layers.GlobalAveragePooling1D()(x_s)                 # (B, 32)
    x_s = layers.Dense(32, activation="relu")(x_s)

    # ------------- Boll branch (軽量Conv→GAP) ----------
    x_b = layers.Conv1D(16, 3, padding="causal", activation="relu")(boll)
    x_b = layers.GlobalAveragePooling1D()(x_b)                 # (B, 16)
    x_b = layers.Dense(32, activation="relu")(x_b)             # (B, 32)

    # ------------- MACD branch (GRU) -------------------
    x_m = layers.GRU(32, return_sequences=False)(macd)         # (B, 32)
    x_m = layers.Dense(32, activation="relu")(x_m)

    # ------------- RSI branch (Conv→GAP) ---------------
    x_r = layers.Conv1D(8, 3, padding="causal", activation="relu")(rsi)
    x_r = layers.GlobalAveragePooling1D()(x_r)                 # (B, 8)
    x_r = layers.Dense(32, activation="relu")(x_r)

    # ------------- 統合 (Add or Concat) ----------------
    fused = layers.Add()([x_p, x_s, x_b, x_m, x_r])            # (B, 32)
    # ─ または Concatenate の場合は次行に差し替え
    # fused = layers.Concatenate()([x_p, x_s, x_b, x_m, x_r])  # (B, 160)

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
