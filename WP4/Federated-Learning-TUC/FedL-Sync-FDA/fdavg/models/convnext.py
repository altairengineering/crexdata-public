from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtLarge, ConvNeXtXLarge, ConvNeXtBase

import os

script_dir = os.path.dirname(os.path.realpath(__file__))
convenet_dir = 'cifar100_imagenet/'


ConvNeXtBase_weight_file = os.path.normpath(
    os.path.join(
        script_dir, f'{convenet_dir}/ConvNeXtBase.04_acc_0.61_val_acc_0.57.weights.h5'
    )
)

ConvNeXtLarge_weight_file = os.path.normpath(
    os.path.join(
        script_dir, f'{convenet_dir}/ConvNeXtLarge.05_acc_0.64_val_acc_0.62.weights.h5'
    )
)

ConvNeXtXLarge_weight_file = os.path.normpath(
    os.path.join(
        script_dir, f'{convenet_dir}/ConvNeXtXLarge.05_acc_0.64_val_acc_0.63.weights.h5'
    )
)


class ConvNeXt:
    def __init__(self, name, input_shape=(32, 32, 3), classes=100):

        base_model, weight_file = None, None

        if name == 'ConvNeXtBase':
            base_model = ConvNeXtBase(include_top=False, weights='imagenet', input_shape=input_shape)
            weight_file = ConvNeXtBase_weight_file

        if name == 'ConvNeXtLarge':
            base_model = ConvNeXtLarge(include_top=False, weights='imagenet', input_shape=input_shape)
            weight_file = ConvNeXtLarge_weight_file

        if name == 'ConvNeXtXLarge':
            base_model = ConvNeXtXLarge(include_top=False, weights='imagenet', input_shape=input_shape)
            weight_file = ConvNeXtXLarge_weight_file

        base_model.trainable = True

        inputs = tf.keras.Input(shape=input_shape)

        # The base model contains batchnorm layers. We want to keep them in inference mode
        # See https://keras.io/guides/transfer_learning/
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(classes, activation='softmax')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)

        self.model.load_weights(weight_file)

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


def get_compiled_and_built_convnext(name, cnn_batch_input, optimizer_fn):
    # TODO: The `build` method gets replaced by `.load_weights` but this function stops making sense this way.
    # TODO: (when time allows) fIx this approach issue, it is confusing.

    convnet = ConvNeXt(name=name)

    convnet.compile(
        optimizer=optimizer_fn(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # we have softmax
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    return convnet
