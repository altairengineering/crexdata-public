from tensorflow.keras import layers, models
import tensorflow as tf

""" 
Implementation from https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py

DenseNet, not pre-trained, specifically for the CIFAR-10 datasets. 

Note:
    - Beforehand preprocessing on input is assumed using `tensorflow.keras.applications.densenet.preprocess_input`.
    - We assume NHWC input data format (which we then handle internally).

Deviations from original keras implementation:
    1) We add dropout layers with rate=0.2 as suggested by Huang et. al, 2016 for training on CIFAR-10
    2) We adopt `he normal` weight-initialization He et al., 2015 as suggested by Huang et. al., 2016
    3) We adopt NCHW data format (channels first). The input is expected to be in NHWC format which we then transform
        to NCHW format. This is due to layout optimization error/explanation of in
        https://github.com/tensorflow/tensorflow/issues/34499#issuecomment-652316457
    4) We do not put `weight_decay=1e-4` in the `optimizers.SGD` but rather equivalently replicate it by putting
        `regularizers.L2(1e-4)` on the weights (kernel) of the `Conv2D` and `Dense` layers. This is because in TF 2.7
        `weight_decay` is not available in the optimizer.
"""


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    bn_axis = 1  # For NCHW format : (batch_size, channels, height, width)

    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=name + '_bn'
    )(x)

    x = layers.Activation('relu', name=name + '_relu')(x)

    x = layers.Conv2D(
        int(x.shape[bn_axis] * reduction),
        1,
        kernel_initializer='he_normal',
        use_bias=False,
        name=name + '_conv',
        data_format='channels_first',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x)

    x = layers.AveragePooling2D(
        2,
        strides=2,
        name=name + '_pool',
        data_format='channels_first'
    )(x)

    return x


def conv_block(x, growth_rate, name):

    bn_axis = 1  # For NCHW format : (batch_size, channels, height, width)

    x1 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=name + '_0_bn'
    )(x)

    x1 = layers.Activation(
        'relu',
        name=name + '_0_relu'
    )(x1)

    x1 = layers.Conv2D(
        4 * growth_rate,
        1,
        use_bias=False,
        kernel_initializer='he_normal',
        name=name + '_1_conv',
        data_format='channels_first',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x1)

    x1 = layers.Dropout(
        0.2,
        name=name + '_1_dropout'
    )(x1)  # Add dropout 0.2 after convolution as Huang et. al suggest for Cifar-10

    x1 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=name + '_1_bn'
    )(x1)

    x1 = layers.Activation(
        'relu',
        name=name + '_1_relu'
    )(x1)

    x1 = layers.Conv2D(
        growth_rate,
        3,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        name=name + '_2_conv',
        data_format='channels_first',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x1)

    x1 = layers.Dropout(
        0.2,
        name=name + '_2_dropout'
    )(x1)  # Add dropout 0.2 after convolution as Huang et. al suggest for Cifar-10

    x = layers.Concatenate(
        axis=bn_axis,
        name=name + '_concat'
    )([x, x1])

    return x


def dense_net_fn(blocks, input_shape, classes):
    # Determine proper input shape
    img_input = layers.Input(shape=input_shape)

    bn_axis = 1  # For NCHW format : (batch_size, channels, height, width)

    x_nchw = tf.transpose(img_input, [0, 3, 1, 2])  # Transform to NCHW format

    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)),
        data_format='channels_first'
    )(x_nchw)

    x = layers.Conv2D(
        64,
        7,
        strides=2,
        use_bias=False,
        kernel_initializer='he_normal',
        name='conv1/conv',
        data_format='channels_first',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x)

    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name='conv1/bn'
    )(x)

    x = layers.Activation(
        'relu',
        name='conv1/relu'
    )(x)

    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)),
        data_format='channels_first'
    )(x)

    x = layers.MaxPooling2D(
        3,
        strides=2,
        name='pool1',
        data_format='channels_first'
    )(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name='bn'
    )(x)

    x = layers.Activation(
        'relu',
        name='relu'
    )(x)

    x = layers.GlobalAveragePooling2D(
        name='avg_pool',
        data_format='channels_first'
    )(x)

    x = layers.Dense(
        classes,
        kernel_initializer='he_normal',
        activation='softmax',
        name='fc10',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x)

    inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = models.Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = models.Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = models.Model(inputs, x, name='densenet201')
    else:
        model = models.Model(inputs, x, name='densenet')

    return model


class DenseNet:
    def __init__(self, name, input_shape=(32, 32, 3), classes=10):

        self.model = None

        if name == 'DenseNet121':
            self.model = dense_net_fn([6, 12, 24, 16], input_shape, classes)
        if name == 'DenseNet169':
            self.model = dense_net_fn([6, 12, 32, 32], input_shape, classes)
        if name == 'DenseNet201':
            self.model = dense_net_fn([6, 12, 48, 32], input_shape, classes)

    def __getattr__(self, name):
        """
        Automatically delegate method calls to the underlying Keras model.
        This ensures that the custom class supports all methods of the
        Keras model without having to define each one explicitly.
        """
        return getattr(self.model, name)

    def step(self, batch):
        x_batch, y_batch = batch
        return self.train_on_batch(x=x_batch, y=y_batch)

    def train(self, dataset):
        for batch in dataset:
            self.step(batch)

    def set_trainable_variables(self, trainable_vars):
        for model_var, var in zip(self.trainable_variables, trainable_vars):
            model_var.assign(var)

    def set_non_trainable_variables(self, non_trainable_vars):
        """
        Set the model's non-trainable variables.

        Args:
        - non_trainable_vars (list of tf.Tensor): A list of tensors representing the non-trainable variables to be set.

        This method sets each of the model's trainable variables to the corresponding tensor in `non_trainable_vars`.
        """
        for model_var, var in zip(self.non_trainable_variables, non_trainable_vars):
            model_var.assign(var)

    @tf.function
    def trainable_vars_as_vector(self):
        return tf.concat([tf.reshape(var, [-1]) for var in self.trainable_variables], axis=0)

    def per_layer_trainable_vars_as_vector(self):
        layer_vectors = [
            tf.concat([tf.reshape(var, [-1]) for var in layer.trainable_weights], axis=0)
            for layer in self.layers
            if layer.trainable_weights
        ]

        return layer_vectors

    def set_layer_weights(self, layer_i, weights):
        for model_var, var in zip(self.layers[layer_i].trainable_weights, weights):
            model_var.assign(var)

    def get_trainable_layers_indices(self):
        trainable_layers_idx = [
            i for i, layer in enumerate(self.layers)
            if layer.trainable_weights
        ]

        return trainable_layers_idx


def get_compiled_and_built_densenet(name, cnn_batch_input, optimizer_fn):
    densenet = DenseNet(name)

    densenet.compile(
        optimizer=optimizer_fn(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # we have softmax
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    densenet.build(cnn_batch_input)

    return densenet
