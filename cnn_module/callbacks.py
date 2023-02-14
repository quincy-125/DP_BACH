import tensorflow as tf
import os

# from tf_explain.callbacks.grad_cam import GradCAMCallback


class CallBacks:
    def __init__(
        self, learning_rate=0.01, log_dir=None, optimizer=None, validation_data=None
    ):
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.validation_data = validation_data
        self.callbacks = self.get_callbacks()

    def _get_tb(self):
        return tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=False,
            update_freq="epoch",
            write_images=False,
        )

    def _get_cp(self):
        # ,save_best_only=True,monitor='val_loss'
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.log_dir, "cp-{epoch:04d}.ckpt"),
            verbose=1,
            save_weights_only=True,
            save_frequency=1,
        )
        # save_weights_only=True,period=1,save_best_only=True) #save_best_only=True)

    @staticmethod
    def _get_es():
        # level3(svs):Patience=7, level1 patience=5 monitor='train_loss' 'val_acc'
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            verbose=1,
            patience=3,
            mode="min",
            restore_best_weights=True,
        )

    def _get_grad_cam(self):
        output_dir = os.path.join(self.log_dir, "output_gradcam")
        os.makedirs(output_dir, exist_ok=True)
        GradCAMCallback(
            validation_data=self.validation_data,
            layer_name="conv5_block3_out",
            class_index=0,
            output_dir=output_dir,
        )
        # return None

    def get_callbacks(self):
        # self._get_grad_cam()
        return [self._get_tb(), self._get_cp(), self._get_es()]
