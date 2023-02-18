import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import l2_normalize
from layers import LosslessTripletLossLayer

tf.keras.backend.clear_session()


def custom(input_shape, embeddingsize):
    # Convolutional Neural Network
    # https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352
    network = Sequential()
    network.add(
        Conv2D(
            128,
            (7, 7),
            activation="relu",
            input_shape=input_shape,
            kernel_initializer="he_uniform",
            kernel_regularizer=l2(2e-4),
        )
    )
    network.add(MaxPooling2D())
    network.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            kernel_regularizer=l2(2e-4),
        )
    )
    network.add(MaxPooling2D())
    network.add(
        Conv2D(
            256,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            kernel_regularizer=l2(2e-4),
        )
    )
    network.add(Flatten())
    network.add(
        Dense(
            4096,
            activation="relu",
            kernel_regularizer=l2(1e-3),
            kernel_initializer="he_uniform",
        )
    )
    network.add(
        Dense(
            embeddingsize,
            activation=None,
            kernel_regularizer=l2(1e-3),
            kernel_initializer="he_uniform",
        )
    )

    # Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: l2_normalize(x, axis=-1)))
    return network

    # Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))


class GetModel:
    def __init__(
        self,
        model_name=None,
        img_size=256,
        embedding_size=128,
        weights="imagenet",
        retrain=True,
        num_layers=None,
        margin=0.2,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.embedding_size = embedding_size
        self.weights = weights
        self.num_layers = num_layers
        self.model, self.preprocess = self.__get_model_and_preprocess(retrain)
        self.margin = margin

    def __get_model_and_preprocess(self, retrain):
        if retrain is True:
            include_top = False
        else:
            include_top = True

        input_tensor = Input(shape=(self.img_size, self.img_size, 3))
        weights = self.weights
        IMG_SHAPE = (self.img_size, self.img_size, 3)

        if self.model_name == "DenseNet121":
            model = tf.keras.applications.DenseNet121(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.model_name == "DenseNet169":
            model = tf.keras.applications.DenseNet169(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.model_name == "DenseNet201":
            model = tf.keras.applications.DenseNet201(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.model_name == "InceptionResNetV2":
            model = tf.keras.applications.InceptionResNetV2(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input(
                input_tensor
            )

        elif self.model_name == "InceptionV3":
            model = tf.keras.applications.InceptionV3(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.inception_v3.preprocess_input(
                input_tensor
            )

        elif self.model_name == "MobileNet":
            model = tf.keras.applications.MobileNet(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.mobilenet.preprocess_input(input_tensor)

        elif self.model_name == "MobileNetV2":
            model = tf.keras.applications.MobileNetV2(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input(
                input_tensor
            )

        elif self.model_name == "NASNetLarge":
            model = tf.keras.applications.NASNetLarge(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.nasnet.preprocess_input(input_tensor)

        elif self.model_name == "NASNetMobile":
            model = tf.keras.applications.NASNetMobile(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.nasnet.preprocess_input(input_tensor)

        elif self.model_name == "ResNet50":
            model = tf.keras.applications.ResNet50(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.resnet50.preprocess_input(input_tensor)

        elif self.model_name == "VGG16":
            model = tf.keras.applications.VGG16(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.vgg16.preprocess_input(input_tensor)

        elif self.model_name == "VGG19":
            model = tf.keras.applications.VGG19(
                weights=weights,
                include_top=include_top,
                input_tensor=input_tensor,
                input_shape=IMG_SHAPE,
            )
            preprocess = tf.keras.applications.vgg19.preprocess_input(input_tensor)

        elif self.model_name == "custom":
            model = custom(IMG_SHAPE, self.embedding_size)
            preprocess = None
        else:
            raise AttributeError(
                "{} not found in available models".format(self.model_name)
            )

        # Add a global average pooling and change the output size to our number of embedding nodes
        if self.model_name != "custom":
            x = model.output
            x = Flatten()(x)
            # x = Dense(4096, activation='relu', activity_regularizer=tf.keras.regularizers.l2(1e-3),kernel_initializer='he_uniform')(x)
            out = Dense(
                self.embedding_size, activity_regularizer=tf.keras.regularizers.l2()
            )(x)
            conv_model = Model(inputs=input_tensor, outputs=out)
            # Now check to see if we are retraining all but the head, or deeper down the stack
            num_trainable_layers = min(self.num_layers, conv_model.layers.__len__()) - 1
            num_nontrainable_layers = conv_model.layers.__len__() - num_trainable_layers
            for i in range(
                conv_model.layers.__len__() - 4
            ):  # Making sure I have at least the last few training always
                if i < num_nontrainable_layers:
                    conv_model.layers[i].trainable = False
                else:
                    conv_model.layers[i].trainable = True

            print(
                "Found {} trainable and {} non_trainable layers".format(
                    num_trainable_layers + 4, num_nontrainable_layers
                )
            )
        else:
            conv_model = model
        # if retrain is True:
        #    x = tf.keras.layers.Dropout(rate=0.2)(x)
        return conv_model, preprocess

    def get_optimizer(self, name, lr=0.001):
        if name == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
        elif name == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif name == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif name == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=lr)
        elif name == "Ftrl":
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr)
        elif name == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        elif name == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif name == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            raise AttributeError("{} not found in available optimizers".format(name))

        return optimizer

    def build_model(self):
        return self.model


def build_triplet_model(patch_size, model, margin=0.2, color_channels=3):
    input_shape = (patch_size, patch_size, color_channels)

    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    # Generate the encodings (feature vectors) for the three images
    encoded_a = model(anchor_input)
    encoded_p = model(positive_input)
    encoded_n = model(negative_input)

    loss_layer = LosslessTripletLossLayer(alpha=margin, name="triplet_loss_layer")(
        [encoded_a, encoded_p, encoded_n]
    )

    # Connect the inputs with the outputs
    network_train = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer
    )

    # return the model
    return network_train
