import tensorflow as tf
import os


class CallBacks:
    def __init__(self, learning_rate=0.01, log_dir=None, optimizer=None):
        super().__init__()
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.callbacks = self.get_callbacks()

    def _getTB(self):
        return tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=False,
            update_freq="epoch",
            write_images=False,
        )

    def _getCP(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.log_dir, "cp-{epoch:04d}.ckpt"),
            verbose=1,
            save_weights_only=True,
            save_frequency=1,
            save_best_only=True,
            monitor="val_accuracy",
        )

    # Uncomment this when fixed in TF                 load_weights_on_restart=True)

    def _getES(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            verbose=1,
            patience=2,
            mode="min",
            restore_best_weights=True,
        )

    def get_callbacks(self):
        return [self._getTB(), self._getCP(), self._getES()]
