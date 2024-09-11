import tensorflow as tf
import numpy as np
from fdavg.data.preprocessing import (create_unbiased_federated_data, create_biased_federated_data,
                                      prepare_federated_data, create_one_label_biased_federated_data,
                                      create_multi_label_biased_federated_data)
from functools import partial
import os

CIFAR100_CNN_BATCH_INPUT = (None, 32, 32, 3)  # CIFAR-100 dataset (None is used for batch size, as it varies)
CIFAR100_N_TRAIN = 50_000

script_dir = os.path.dirname(os.path.realpath(__file__))
cifar100_dir = 'data/cifar100/'
cifar100_part1_data = os.path.normpath(os.path.join(script_dir, f'{cifar100_dir}/cifar100_part1.npz'))
cifar100_part2_data = os.path.normpath(os.path.join(script_dir, f'{cifar100_dir}/cifar100_part2.npz'))


def load_cifar100_from_local_npz():

    with np.load(cifar100_part1_data) as data1:
        with np.load(cifar100_part2_data) as data2:
            X_train = np.concatenate((data1['X_train'], data2['X_train']))
            y_train = np.concatenate((data1['y_train'], data2['y_train']))
            X_test = np.concatenate((data1['X_test'], data2['X_test']))
            y_test = np.concatenate((data1['y_test'], data2['y_test']))

            return (X_train, y_train), (X_test, y_test)


def cifar100_load_data():
    """
    Load the CIFAR-100 dataset without any preprocessing (EfficientNet has incorporated processing)

    This function loads the CIFAR-10 dataset using Keras's built-in dataset API.
    It normalizes the pixel values of the images by dividing them by 255.0.

    Returns:
    - X_train (numpy.ndarray): The training data, a 3D array of shape (num_samples, 32, 32, 3).
    - y_train (numpy.ndarray): The labels for the training data, a 1D array of shape (num_samples,).
    - X_test (numpy.ndarray): The test data, a 3D array of shape (num_samples, 32, 32, 3).
    - y_test (numpy.ndarray): The labels for the test data, a 1D array of shape (num_samples,).
    """
    (X_train, y_train), (X_test, y_test) = load_cifar100_from_local_npz()

    return X_train, y_train, X_test, y_test


def cifar100_load_federated_data(num_clients, batch_size, num_steps_until_rtc_check, bias=None, seed=None):
    """
    Load and prepare the federated CIFAR-10 dataset with an optional bias.

    This function uses the CIFAR-10 dataset and prepares it for federated learning. If a bias value
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

    X_train, y_train, X_test, y_test = cifar100_load_data()

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    create_federated_data_fn = None

    if not bias:
        create_federated_data_fn = create_unbiased_federated_data

    elif bias >= 0:
        create_federated_data_fn = partial(create_biased_federated_data, bias=bias)

    elif bias == -1:
        create_federated_data_fn = partial(create_one_label_biased_federated_data, biased_label=6)

    elif bias == -2:
        create_federated_data_fn = partial(create_one_label_biased_federated_data, biased_label=8)

    elif bias == -3:
        create_federated_data_fn = partial(create_multi_label_biased_federated_data, biased_labels_list=list(range(1, 21)))

    federated_ds = prepare_federated_data(
        federated_dataset=create_federated_data_fn(X_train, y_train, num_clients),
        batch_size=batch_size,
        num_steps_until_rtc_check=num_steps_until_rtc_check,
        seed=seed
    )

    return federated_ds, test_ds






