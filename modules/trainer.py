import tensorflow as tf


class GradualDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """学習率を徐々に減衰させるスケジューラー"""

    def __init__(self, initial_lr, final_lr, decay_steps):
        super().__init__()
        self.decay_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            end_learning_rate=final_lr,
            power=1.0
        )

    def __call__(self, step):
        return self.decay_fn(step)


class Trainer:
    """簡素化された学習管理クラス"""

    def __init__(self,
                 model,
                 train_data,
                 valid_data,
                 test_data,
                 learning_rate_initial=1e-4,
                 learning_rate_final=1e-5,
                 switch_epoch=50,
                 max_epochs=1000,
                 batch_size=1024,
                 early_stop_patience=100):
        self.model = model
        self.train_x, self.train_y = train_data
        self.valid_x, self.valid_y = valid_data
        self.test_x, self.test_y = test_data

        self.learning_rate_schedule = GradualDecaySchedule(
            learning_rate_initial, learning_rate_final, switch_epoch
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_schedule)

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience

        # ベストモデル情報
        self.best_weights = None
        self.best_val_loss = float("inf")

    def run(self):
        """model.fitだけで学習・early stopping"""
        self.model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stop_patience,
            restore_best_weights=True
        )

        history = self.model.fit(
            self.train_x,
            self.train_y,
            validation_data=(self.valid_x, self.valid_y),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        # 学習後、ベスト重みを保持
        self.best_weights = self.model.get_weights()

        # テストデータで最終評価
        test_loss, test_acc = self.model.evaluate(
            self.test_x, self.test_y, batch_size=self.batch_size, verbose=0)
        print(
            f"[INFO] Test result -> loss={test_loss:.6f}, acc={test_acc:.6f}")

        valid_loss, valid_acc = self.model.evaluate(
            self.valid_x, self.valid_y, batch_size=self.batch_size, verbose=0)

        return valid_loss, valid_acc, test_loss, test_acc

    def save_best_weights(self, filepath):
        """ベストモデルの重みを保存する"""
        if self.best_weights is None:
            print("[WARN] No best weights found. Not saving anything.")
            return
        self.model.set_weights(self.best_weights)
        self.model.save_weights(filepath)
        print(f"[INFO] Saved best weights to: {filepath}")
