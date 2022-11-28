import tensorflow as tf


class NG_Att_Net(tf.keras.Model):
    def __init__(
        self,
        dim_features=1024,
        dim_compress_features=512,
        n_hidden_units=256,
        n_class=2,
        dropout=False,
        dropout_rate=0.25,
    ):
        super(NG_Att_Net, self).__init__()
        self.dim_features = dim_features
        self.dim_compress_features = dim_compress_features
        self.n_hidden_units = n_hidden_units
        self.n_class = n_class
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.compression_model = tf.keras.models.Sequential()
        self.model = tf.keras.models.Sequential()

        self.fc_compress_layer = tf.keras.layers.Dense(
            units=dim_compress_features,
            activation="relu",
            input_shape=(dim_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Fully_Connected_Layer",
        )

        self.compression_model.add(self.fc_compress_layer)

        self.att_layer1 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="linear",
            input_shape=(dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_layer1",
        )

        self.att_layer2 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="tanh",
            input_shape=(dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_Layer2",
        )

        self.att_layer3 = tf.keras.layers.Dense(
            units=n_class,
            activation="linear",
            input_shape=(n_hidden_units,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_Layer3",
        )

        self.model.add(self.att_layer1)
        self.model.add(self.att_layer2)

        if dropout:
            self.model.add(tf.keras.layers.Dropout(dropout_rate, name="Dropout_Layer"))

        self.model.add(self.att_layer3)

    def att_model(self):
        attention_model = [self.compression_model, self.model]
        return attention_model

    def call(self, img_features):
        h = list()
        A = list()

        for i in img_features:
            c_imf = self.att_model()[0](i)
            h.append(c_imf)

        for j in h:
            a = self.att_model()[1](j)
            A.append(a)
        return h, A


class G_Att_Net(tf.keras.Model):
    def __init__(
        self,
        dim_features=1024,
        dim_compress_features=512,
        n_hidden_units=256,
        n_class=2,
        dropout=False,
        dropout_rate=0.25,
    ):
        super(G_Att_Net, self).__init__()
        self.dim_features = dim_features
        self.dim_compress_features = dim_compress_features
        self.n_hidden_units = n_hidden_units
        self.n_class = n_class
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.compression_model = tf.keras.models.Sequential()
        self.model_v = tf.keras.models.Sequential()
        self.model_u = tf.keras.models.Sequential()
        self.model = tf.keras.models.Sequential()

        self.fc_compress_layer = tf.keras.layers.Dense(
            units=dim_compress_features,
            activation="relu",
            input_shape=(dim_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Fully_Connected_Layer",
        )

        self.compression_model.add(self.fc_compress_layer)

        self.att_v_layer1 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="linear",
            input_shape=(dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_V_Layer1",
        )

        self.att_v_layer2 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="tanh",
            input_shape=(dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_V_Layer2",
        )

        self.att_u_layer1 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="linear",
            input_shape=(dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_U_Layer1",
        )

        self.att_u_layer2 = tf.keras.layers.Dense(
            units=n_hidden_units,
            activation="sigmoid",
            input_shape=(dim_compress_features,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_U_Layer2",
        )

        self.att_layer_f = tf.keras.layers.Dense(
            units=n_class,
            activation="linear",
            input_shape=(n_hidden_units,),
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            name="Attention_Gated_Final_Layer",
        )

        self.model_v.add(self.att_v_layer1)
        self.model_v.add(self.att_v_layer2)

        self.model_u.add(self.att_u_layer1)
        self.model_u.add(self.att_u_layer2)

        if dropout:
            self.model_v.add(
                tf.keras.layers.Dropout(dropout_rate, name="Dropout_V_Layer")
            )
            self.model_u.add(
                tf.keras.layers.Dropout(dropout_rate, name="Dropout_U_Layer")
            )

        self.model.add(self.att_layer_f)

    def att_model(self):
        attention_model = [
            self.compression_model,
            self.model_v,
            self.model_u,
            self.model,
        ]
        return attention_model

    def call(self, img_features):
        h = list()
        A = list()

        for i in img_features:
            c_imf = self.att_model()[0](i)
            h.append(c_imf)

        for j in h:
            att_v_output = self.att_model()[1](j)
            att_u_output = self.att_model()[2](j)
            att_input = tf.math.multiply(att_v_output, att_u_output)
            a = self.att_model()[3](att_input)
            A.append(a)

        return h, A
