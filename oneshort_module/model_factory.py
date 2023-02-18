import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Lambda,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
    Dropout,
)
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import abs

W_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
W_init_fc = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)
b_init = tf.keras.initializers.TruncatedNormal(mean=0.05, stddev=0.01)


def get_emb_vec(IMG_SHAPE):
    IMG_SHAPE = (224, 224, 3)
    input_shape = IMG_SHAPE

    W_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    W_init_fc = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)
    b_init = tf.keras.initializers.TruncatedNormal(mean=0.05, stddev=0.01)
    # model = Sequential()
    # model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
    #                kernel_initializer=W_init, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (7,7), activation='relu',
    #                  kernel_initializer=W_init,
    #                  bias_initializer=b_init, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=W_init,
    #                  bias_initializer=b_init, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=W_init,
    #                  bias_initializer=b_init, kernel_regularizer=l2(2e-4)))
    # model.trainable = True

    model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=IMG_SHAPE
    )
    model.trainable = False
    x = model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    out = Dense(
        128,
        kernel_regularizer=l2(2e-4),
        kernel_initializer=W_init_fc,
        bias_initializer=b_init,
    )(x)
    conv_model = Model(inputs=model.input, outputs=out)
    return conv_model


def custom(IMG_SHAPE):
    input_shape = IMG_SHAPE
    # convnet = Sequential()
    # convnet.add(Conv2D(8,(2,2),activation='relu',input_shape=input_shape,
    #                    kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
    # convnet.add(MaxPooling2D())
    # convnet.add(Conv2D(32,(3,3),activation='relu',
    #                    kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
    # convnet.add(MaxPooling2D())
    # convnet.add(Conv2D(1,(2,2),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
    # return convnet

    model = Sequential()
    model.add(
        Conv2D(
            64,
            (10, 10),
            activation="relu",
            input_shape=input_shape,
            kernel_initializer=W_init,
            kernel_regularizer=l2(2e-4),
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            128,
            (7, 7),
            activation="relu",
            kernel_initializer=W_init,
            bias_initializer=b_init,
            kernel_regularizer=l2(2e-4),
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            128,
            (4, 4),
            activation="relu",
            kernel_initializer=W_init,
            bias_initializer=b_init,
            kernel_regularizer=l2(2e-4),
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            256,
            (4, 4),
            activation="relu",
            kernel_initializer=W_init,
            bias_initializer=b_init,
            kernel_regularizer=l2(2e-4),
        )
    )
    return model


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
        self.model = self.__get_model(retrain)
        self.margin = margin

    def __get_model(self, retrain=True):
        if retrain is True:
            include_top = False
        else:
            include_top = True
        weights = self.weights
        IMG_SHAPE = (self.img_size, self.img_size, 3)
        anchor_input_tensor = Input(shape=IMG_SHAPE, name="anchor_img")
        other_input_tensor = Input(shape=IMG_SHAPE, name="other_img")
        input_tensor = Input(shape=IMG_SHAPE, name="anchor_img")
        if self.model_name == "DenseNet121":
            model = tf.keras.applications.DenseNet121(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "DenseNet169":
            model = tf.keras.applications.DenseNet169(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "DenseNet201":
            model = tf.keras.applications.DenseNet201(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "InceptionResNetV2":
            model = tf.keras.applications.InceptionResNetV2(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "InceptionV3":
            model = tf.keras.applications.InceptionV3(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "MobileNet":
            model = tf.keras.applications.MobileNet(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "MobileNetV2":
            model = tf.keras.applications.MobileNetV2(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "NASNetLarge":
            model = tf.keras.applications.NASNetLarge(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "NASNetMobile":
            model = tf.keras.applications.NASNetMobile(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "ResNet50":
            model = tf.keras.applications.ResNet50(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "ResNet152":
            model = tf.keras.applications.ResNet152(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "VGG16":
            model = tf.keras.applications.VGG16(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "VGG19":
            model = tf.keras.applications.VGG19(
                weights=weights, include_top=include_top, input_shape=IMG_SHAPE
            )

        elif self.model_name == "custom":
            model = custom(IMG_SHAPE)

        elif self.model_name == "two_layer_nn":
            model = two_layer_nn(IMG_SHAPE)
        else:
            raise AttributeError(
                "{} not found in available models".format(self.model_name)
            )

        # model.trainable = True
        # if self.model_name != 'custom':
        # model.trainable = False
        for layer in model.layers:
            layer.trainable = True
        x = model.output
        # if retrain is True:
        x = GlobalAveragePooling2D(name="avg_pool")(x)

        # x = BatchNormalization()(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.5)(x)
        # x = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
        x = Flatten()(x)

        # x = Dense(4096,activation='sigmoid', kernel_regularizer=l2(0.0001), kernel_initializer=W_init_fc, bias_initializer=b_init)(x)
        x = Dropout(0.35)(x)
        x = Dense(
            1024,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
        )(x)
        x = Dropout(0.35)(x)
        x = Dense(
            512,
            activation="sigmoid",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
        )(x)

        # x = Dropout(0.3)(x)
        # , kernel_regularizer=l2(0.0002), kernel_initializer=W_init_fc, bias_initializer=b_init
        out = Dense(self.embedding_size, activation="sigmoid")(x)
        # out = Dense(4096,activation='sigmoid', kernel_initializer=W_init_fc, bias_initializer=b_init)(x)
        # out = Dense(self.classes, kernel_regularizer=regularizers.l2(0.0001), activation='softmax')(x)
        conv_model = Model(inputs=model.input, outputs=out)

        anchor_encoded = conv_model(anchor_input_tensor)
        other_encoded = conv_model(other_input_tensor)
        # Get L1 Distances
        L1_layer = Lambda(lambda tensors: abs(tensors[0] - tensors[1]))
        x = L1_layer([anchor_encoded, other_encoded])
        prediction = Dense(2, activation="sigmoid", bias_initializer=b_init)(x)
        # prediction = Lambda(lambda x: tf.squeeze(x))(prediction)
        siamese_net = Model(
            inputs=[anchor_input_tensor, other_input_tensor], outputs=prediction
        )

        return siamese_net

    def get_optimizer(self, name, lr=0.00006):
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
