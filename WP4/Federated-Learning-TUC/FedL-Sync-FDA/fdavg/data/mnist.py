import tensorflow as tf
import numpy as np
from fdavg.data.preprocessing import (create_unbiased_federated_data, create_biased_federated_data,
                                      prepare_federated_data, create_one_label_biased_federated_data)
from functools import partial
import os

MNIST_CNN_BATCH_INPUT = (None, 28, 28)  # MNIST dataset (None is used for batch size, as it varies)
MNIST_CNN_INPUT_RESHAPE = (28, 28, 1)
MNIST_N_TRAIN = 60_000


script_dir = os.path.dirname(os.path.realpath(__file__))
mnist_dir = 'data/mnist/'
mnist_data = os.path.normpath(os.path.join(script_dir, f'{mnist_dir}/mnist.npz'))


def load_mnist_from_local_npz():
    with np.load(mnist_data) as data:
        return (data['X_train'], data['y_train']), (data['X_test'], data['y_test'])


def mnist_load_data():
    """
    Load the MNIST dataset and normalize the pixel values.

    This function loads the MNIST dataset using Keras's built-in dataset API.
    It normalizes the pixel values of the images by dividing them by 255.0.
    
    Returns:
    - X_train (numpy.ndarray): The training data, a 3D array of shape (num_samples, 28, 28).
    - y_train (numpy.ndarray): The labels for the training data, a 1D array of shape (num_samples,).
    - X_test (numpy.ndarray): The test data, a 3D array of shape (num_samples, 28, 28).
    - y_test (numpy.ndarray): The labels for the test data, a 1D array of shape (num_samples,).
    """
    (X_train, y_train), (X_test, y_test) = load_mnist_from_local_npz()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    return X_train, y_train, X_test, y_test


def mnist_load_federated_data(num_clients, batch_size, num_steps_until_rtc_check, bias=None, seed=None):
    """
    Load and prepare the federated MNIST dataset with an optional bias.

    This function uses the MNIST dataset and prepares it for federated learning. If a bias value
    is provided, the data is sharded among clients in a way that introduces the specified bias.
    If no bias value is given, the data is sharded uniformly across clients. The data is then batched and shuffled.

    Args:
        num_clients (int): The number of clients among which the data should be distributed.
        batch_size (int): The size of the batches to be used.
        num_steps_until_rtc_check (int): The number of batches to take from each client dataset.
        bias (float, optional): The proportion of the data that should be biased. If provided,
            the data will be biased according to this value; otherwise, it will be unbiased.
            The value should be between 0 and 1. Default is None. If -1 given then we create
            one label biasing. See corresponding comments.
        seed (int, optional): The random seed for shuffling the dataset. Default is None.

    Returns:
        tuple: A tuple containing:
            - federated_ds (list of tf.data.Dataset): A list of prepared TensorFlow Dataset objects
              for federated training.
            - test_ds (tf.data.Dataset): A TensorFlow Dataset object for testing.
    """

    X_train, y_train, X_test, y_test = mnist_load_data()

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    create_federated_data_fn = None

    if not bias:
        create_federated_data_fn = create_unbiased_federated_data

    elif bias >= 0:
        create_federated_data_fn = partial(create_biased_federated_data, bias=bias)

    elif bias == -1:
        create_federated_data_fn = partial(create_one_label_biased_federated_data, biased_label=0)

    elif bias == -2:
        create_federated_data_fn = partial(create_one_label_biased_federated_data, biased_label=8)

    federated_ds = prepare_federated_data(
        federated_dataset=create_federated_data_fn(X_train, y_train, num_clients),
        batch_size=batch_size,
        num_steps_until_rtc_check=num_steps_until_rtc_check,
        seed=seed
    )

    return federated_ds, test_ds






