import tensorflow as tf

class LeNet5(tf.keras.Model):
    """
    LeNet-5 model for image classification.

    Attributes:
    - Layers for the LeNet-5 architecture (convolutional, pooling, dense layers).

    Methods:
    - call: Forward pass for the model.
    - step: Compute and apply gradients for one training batch.
    - train: Train the model on a dataset.
    - set_trainable_variables: Set the trainable variables of the model.
    - trainable_vars_as_vector: Return the trainable variables as a 1D tensor.
    """
    def __init__(self, cnn_input_reshape, num_classes):
        """
        Initialize the LeNet-5 model with given input shape and number of output classes.

        Args:
        - cnn_input_reshape (tuple): The shape to which the input should be reshaped. (e.g., (28, 28, 1))
        - num_classes (int): Number of output classes.
        """
        
        super(LeNet5, self).__init__()
        
        self.reshape = tf.keras.layers.Reshape(cnn_input_reshape)
        
        # Layer 1 Conv2D
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same')
        # Layer 2 Pooling Layer
        self.avgpool1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # Layer 3 Conv2D
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')
        # Layer 4 Pooling Layer
        self.avgpool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=120, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(units=84, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs, training=None):
        """
        Forward pass for the model.

        Args:
        - inputs (tf.Tensor): Input tensor (batch of images).
        - training (bool, optional): Whether the forward pass is for training.

        Returns:
        - x (tf.Tensor): Output tensor (batch of class probabilities).
        """
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return x

    def step(self, batch):
        x_batch, y_batch = batch
        return self.train_on_batch(x=x_batch, y=y_batch)

    def train(self, dataset):
        """
        Train the model on an entire dataset.

        Args:
        - dataset (tf.data.Dataset): The dataset on which the model will be trained. 
        """
        for batch in dataset:
            self.step(batch)

    def set_trainable_variables(self, trainable_vars):
        """
        Set the model's trainable variables.

        Args:
        - trainable_vars (list of tf.Tensor): A list of tensors representing the trainable variables to be set.

        This method sets each of the model's trainable variables to the corresponding tensor in `trainable_vars`.
        """
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
        """
        Get the model's trainable variables as a single vector.

        Returns:
        - tf.Tensor: A 1D tensor containing all of the model's trainable variables.
        """
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


def get_compiled_and_built_lenet(cnn_batch_input, cnn_input_reshape, num_classes, optimizer_fn):
    """
    Compile and build a LeNet-5 model.

    Args:
    - cnn_batch_input (tuple): The shape of the input including batch size (e.g., (None, 28, 28)).
    - cnn_input_reshape (tuple): The shape to which the input should be reshaped (e.g., (28, 28, 1)).
    - num_classes (int): Number of output classes.
    - optimizer_fn (callable): function initializing the optimizer

    Returns:
    - cnn (LeNet5): A compiled and built LeNet-5 model.
    """
    cnn = LeNet5(cnn_input_reshape, num_classes)
    
    cnn.compile(
        optimizer=optimizer_fn(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # we have softmax
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    
    cnn.build(cnn_batch_input)
    
    return cnn


# Sequential API - for reference
def sequential_lenet5(cnn_input_reshape, num_classes):
    """
    Args:
    - cnn_input_reshape (tuple): The shape to which the input should be reshaped (e.g., (28, 28, 1)).
    - num_classes (int): Number of output classes.

    Returns:
    - tf.keras.models.Sequential: A LeNet-5 model using the Sequential API.

    Example for MNIST:
        lenet5 = sequential_lenet5((28, 28, 1), 10)
        lenet5.compile(...)
        lenet5.fit(...)
    """
    return tf.keras.models.Sequential([
        # Reshape layer
        tf.keras.layers.Reshape(cnn_input_reshape, input_shape=(28, 28)),  # Example input shape, change as needed
        # Layer 1 Conv2D
        tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same'),
        # Layer 2 Pooling Layer
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        # Layer 3 Conv2D
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
        # Layer 4 Pooling Layer
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        # Flatten
        tf.keras.layers.Flatten(),
        # Layer 5 Dense
        tf.keras.layers.Dense(units=120, activation='tanh'),
        # Layer 6 Dense
        tf.keras.layers.Dense(units=84, activation='tanh'),
        # Layer 7 Dense
        tf.keras.layers.Dense(units=num_classes, activation='softmax')  # Example num_classes=10, change as needed
    ])
