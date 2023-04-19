import tensorflow as tf
import logging
import sys

logging.basicConfig(
    stream=sys.stderr, level="DEBUG", format="%(name)s (%(levelname)s): %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class GetModel:
    def __init__(
        self,
        args,
    ):
        """_summary_

        Args:
            args (_type_): _description_
        """
        self.args = args

    def get_model_and_preprocess(self,):
        """_summary_

        Raises:
            AttributeError: _description_

        Returns:
            _type_: _description_
        """
        input_tensor = tf.keras.layers.Input(shape=(self.args.patch_size, self.args.patch_size, 3))
        img_shape = (self.args.patch_size, self.args.patch_size, 3)

        if self.args.model_name == "DenseNet121":
            model = tf.keras.applications.DenseNet121(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.args.model_name == "DenseNet169":
            model = tf.keras.applications.DenseNet169(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.args.model_name == "DenseNet201":
            model = tf.keras.applications.DenseNet201(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.args.model_name == "InceptionResNetV2":
            model = tf.keras.applications.InceptionResNetV2(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input(
                input_tensor
            )

        elif self.args.model_name == "InceptionV3":
            model = tf.keras.applications.InceptionV3(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.inception_v3.preprocess_input(
                input_tensor
            )

        elif self.args.model_name == "MobileNet":
            model = tf.keras.applications.MobileNet(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.mobilenet.preprocess_input(input_tensor)

        elif self.args.model_name == "MobileNetV2":
            model = tf.keras.applications.MobileNetV2(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input(
                input_tensor
            )

        elif self.args.model_name == "NASNetLarge":
            model = tf.keras.applications.NASNetLarge(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.nasnet.preprocess_input(input_tensor)

        elif self.args.model_name == "NASNetMobile":
            model = tf.keras.applications.NASNetMobile(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.nasnet.preprocess_input(input_tensor)

        elif self.args.model_name == "ResNet50":
            model = tf.keras.applications.ResNet50(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.resnet50.preprocess_input(input_tensor)

        elif self.args.model_name == "ResNet152":
            model = tf.keras.applications.ResNet152(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.resnet.preprocess_input(input_tensor)

        elif self.args.model_name == "VGG16":
            print("Model loaded was VGG16")
            model = tf.keras.applications.VGG16(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.vgg16.preprocess_input(input_tensor)

        elif self.args.model_name == "VGG19":
            model = tf.keras.applications.VGG19(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.vgg19.preprocess_input(input_tensor)

        elif self.args.model_name == "Xception":
            model = tf.keras.applications.Xception(
                weights=self.args.weights,
                include_top=self.args.include_top,
                input_tensor=input_tensor,
                input_shape=img_shape,
            )
            preprocess = tf.keras.applications.xception.preprocess_input(input_tensor)

        else:
            raise AttributeError(
                f"{self.args.model_name} not found in available models"
            )

        # Add a global average pooling and change the output size to our number of classes
        base_model = model

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = tf.keras.layers.Flatten()(x)

        if self.args.dropout_rate != 0.0:
            x = tf.keras.layers.Dropout(self.args.dropout_rate)(x)
        if self.args.l2_reg != -1:
            x = tf.keras.layers.Dense(
                1024,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(self.args.l2_reg),
            )(x)
        else:
            x = tf.keras.layers.Dense(1024, activation="relu")(x)
        if self.args.dropout_rate != 0.0:
            x = tf.keras.layers.Dropout(self.args.dropout_rate)(x)
        if self.args.l2_reg != -1:
            x = tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(self.args.l2_reg),
            )(x)
        else:
            x = tf.keras.layers.Dense(512, activation="relu")(x)

        out = tf.keras.layers.Dense(self.args.classes, activation="softmax")(x)

        base_model.trainable = False

        # Now check to see if we are retraining all but the head, or deeper down the stack
        if self.args.num_layers != -1:
            if self.args.num_layers == 0:
                for layer in base_model.layers:
                    layer.trainable = True
            if self.args.num_layers > 0:
                for layer in base_model.layers[: self.args.num_layers]:
                    layer.trainable = False
                for layer in base_model.layers[self.args.num_layers :]:
                    layer.trainable = True

        conv_model = tf.keras.Model(inputs=input_tensor, outputs=out)

        return conv_model, preprocess

    def get_loss(self,):
        """_summary_

        Raises:
            AttributeError: _description_

        Returns:
            _type_: _description_
        """
        if self.args.loss_function == "BinaryCrossentropy":
            return tf.keras.losses.BinaryCrossentropy()
        elif self.args.loss_function == "SparseCategoricalCrossentropy":
            print("Loss is SparseCategoricalCrossentropy")
            return tf.keras.losses.SparseCategoricalCrossentropy()
        elif self.args.loss_function == "CategoricalCrossentropy":
            return tf.keras.losses.CategoricalCrossentropy()
        elif self.args.loss_function == "Hinge":
            return tf.keras.losses.Hinge()
        else:
            raise AttributeError(f"{self.args.loss_function} as a loss function is not yet coded")

    def get_optimizer(self,):
        if self.args.optimizer_name == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.args.lr)
        elif self.args.optimizer_name == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.args.lr)
        elif self.args.optimizer_name == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)
        elif self.args.optimizer_name == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=self.args.lr)
        elif self.args.optimizer_name == "Ftrl":
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=self.args.lr)
        elif self.args.optimizer_name == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.args.lr)
        elif self.args.optimizer_name == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.args.lr)
        elif self.args.optimizer_name == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.args.lr)
        else:
            raise AttributeError(
                f"{self.args.optimizer_name} not found in available optimizers"
            )
        return optimizer

    def compile_model(self,):
        model, preprocess = self.get_model_and_preprocess()

        # Define the trainable model
        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.get_loss(),
            metrics=[
                tf.keras.metrics.AUC(curve='PR', num_thresholds=10, name='PR'),
                tf.keras.metrics.AUC(name='AUC'),
                tf.keras.metrics.AUC(curve='PR',name='PR'),
                tf.keras.metrics.Accuracy(name='accuracy'),
                tf.keras.metrics.CategoricalAccuracy(name='CategoricalAccuracy'),
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn'),
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
            ],
        )

        return model