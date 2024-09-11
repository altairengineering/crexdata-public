import tensorflow as tf
import numpy as np


def prepare_federated_data(federated_dataset, batch_size, num_steps_until_rtc_check, seed=None):
    """
    Prepare federated data by sharding the original training dataset across multiple clients.

    https://cs230.stanford.edu/blog/datapipeline/#best-practices

    Args:
        federated_dataset (list of tf.data.Dataset): A list of datasets for each client.
        batch_size (int): The size of the batches to be used.
        num_steps_until_rtc_check (int): The number of batches to take from each client dataset.
        seed (int, optional): The random seed for shuffling the dataset. Default is None.

    Returns:
        federated_dataset_prepared (list of tf.data.Dataset): A list of prepared TensorFlow Dataset objects,
            one for each client dataset. The datasets are batched, shuffled.
            Each element of a prepared client dataset is a batch containing:
            - A 3D tensor of shape (batch_size, 28, 28), representing a batch of grayscale images.
            - A 1D tensor of shape (batch_size,), representing the labels of the images in the batch.
    """

    def process_client_dataset(_client_dataset, _batch_size, _num_steps_until_rtc_check, _seed):
        shuffle_size = _client_dataset.cardinality()  # Uniform shuffling
        return _client_dataset.shuffle(shuffle_size, seed=_seed).repeat().batch(_batch_size) \
            .take(_num_steps_until_rtc_check)

    federated_dataset_prepared = [
        process_client_dataset(client_dataset, batch_size, num_steps_until_rtc_check, seed)
        for client_dataset in federated_dataset
    ]
    return federated_dataset_prepared


def create_unbiased_federated_data(X_train, y_train, num_clients):
    """
    Create federated data by equally distributing the dataset across multiple clients.

    This function shards the given training dataset uniformly across the specified number of clients.

    Args:
        X_train (numpy.ndarray): The training data features.
        y_train (numpy.ndarray): The training data labels.
        num_clients (int): The number of clients among which the data should be distributed.

    Returns:
        list of tf.data.Dataset: A list of TensorFlow Dataset objects. Each dataset in the list corresponds to
        the data shard for a client. The order of the datasets in the list corresponds to the order of the clients.

    Example:
        >>> X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y_train = np.array([0, 1, 0, 1])
        >>> num_clients = 2
        >>> federated_ds = create_unbiased_federated_data(X_train, y_train, num_clients)
        >>> len(federated_ds)
        2
    """
    X_train_unbiased_lst = np.array_split(X_train, num_clients)
    y_train_unbiased_lst = np.array_split(y_train, num_clients)

    unbiased_federated_dataset = [
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        for X_train, y_train in zip(X_train_unbiased_lst, y_train_unbiased_lst)
    ]

    return unbiased_federated_dataset


def create_biased_federated_data(X_train, y_train, num_clients, bias):
    """
    Create federated data with a specified bias across multiple clients.

    This function shards the given training dataset among clients in a way that introduces a specified bias.
    The bias is applied by sorting the labels and distributing them across the clients evenly. Each clients'
    dataset is composed of `bias`% of biased samples.

    Args:
        X_train (numpy.ndarray): The training data features.
        y_train (numpy.ndarray): The training data labels.
        num_clients (int): The number of clients among which the data should be distributed.
        bias (float): The proportion of the data that should be biased. A value between 0 and 1.

    Returns:
        list of tf.data.Dataset: A list of TensorFlow Dataset objects. Each dataset in the list corresponds to
        the data shard for a client. The order of the datasets in the list corresponds to the order of the clients.

    Example:
        >>> X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y_train = np.array([0, 1, 0, 1])
        >>> num_clients = 2
        >>> bias = 0.5
        >>> federated_ds = create_biased_federated_data(X_train, y_train, num_clients, bias)
        >>> it = iter(federated_ds[0])
        >>> next(it)
        (<tf.Tensor: shape=(2,), dtype=int64, numpy=array([1, 2])>, <tf.Tensor: shape=(), dtype=int64, numpy=0>)
        >>> next(it)
        (<tf.Tensor: shape=(2,), dtype=int64, numpy=array([5, 6])>, <tf.Tensor: shape=(), dtype=int64, numpy=0>)
        >>> it = iter(federated_ds[1])
        >>> next(it)
        (<tf.Tensor: shape=(2,), dtype=int64, numpy=array([3, 4])>, <tf.Tensor: shape=(), dtype=int64, numpy=1>)
        >>> next(it)
        (<tf.Tensor: shape=(2,), dtype=int64, numpy=array([7, 8])>, <tf.Tensor: shape=(), dtype=int64, numpy=1>)
    """

    n = len(y_train)
    biased_n = int(n * bias)

    X_train_biased = X_train[:biased_n]
    y_train_biased = y_train[:biased_n]

    X_train_unbiased = X_train[biased_n:]
    y_train_unbiased = y_train[biased_n:]

    sorted_indices = y_train_biased.argsort()

    X_train_sorted = X_train_biased[sorted_indices]
    y_train_sorted = y_train_biased[sorted_indices]

    X_train_biased_lst = np.array_split(X_train_sorted, num_clients)
    y_train_biased_lst = np.array_split(y_train_sorted, num_clients)

    X_train_unbiased_lst = np.array_split(X_train_unbiased, num_clients)
    y_train_unbiased_lst = np.array_split(y_train_unbiased, num_clients)

    biased_federated_dataset = []

    client_data_gen = zip(X_train_biased_lst, y_train_biased_lst, X_train_unbiased_lst, y_train_unbiased_lst)

    for X_biased, y_biased, X_unbiased, y_unbiased in client_data_gen:

        X_train = np.concatenate((X_biased, X_unbiased))
        y_train = np.concatenate((y_biased, y_unbiased))

        biased_federated_dataset.append(
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
        )

    return biased_federated_dataset


def create_one_label_biased_federated_data(X_train, y_train, num_clients, biased_label):
    """
    Create non-iid federated data with a specific label (biased_label) completely non-uniformly distributed (potentially
    whole samples of said label will go on a few clients only). Almost equal cardinality of each clients' dataset.
    Rest of dataset is iid (without the zero label examples).

    Args:
        X_train (numpy.ndarray): The training data features.
        y_train (numpy.ndarray): The training data labels.
        num_clients (int): The number of clients among which the data should be distributed.

    Returns:
        list of tf.data.Dataset: A list of TensorFlow Dataset objects. Each dataset in the list corresponds to
        the data shard for a client. The order of the datasets in the list corresponds to the order of the clients.
    """

    X_train_zeros = X_train[y_train == biased_label]
    y_train_zeros = y_train[y_train == biased_label]

    X_train_rest = X_train[y_train != biased_label]
    y_train_rest = y_train[y_train != biased_label]

    X_train_one_label_biased = np.concatenate((X_train_zeros, X_train_rest))
    y_train_one_label_biased = np.concatenate((y_train_zeros, y_train_rest))

    X_train_one_label_biased_lst = np.array_split(X_train_one_label_biased, num_clients)
    y_train_one_label_biased_lst = np.array_split(y_train_one_label_biased, num_clients)

    one_label_biased_federated_dataset = []

    for X_train, y_train in zip(X_train_one_label_biased_lst, y_train_one_label_biased_lst):
        one_label_biased_federated_dataset.append(
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
        )

    return one_label_biased_federated_dataset


def create_multi_label_biased_federated_data(X_train, y_train, num_clients, biased_labels_list):
    """
    Create non-iid federated data with specific labels (biased_labels_list) completely non-uniformly distributed
    (potentially whole samples of said labels will go on a few clients only). Almost equal cardinality of each client's dataset.
    Rest of the dataset is iid (without the specified labels examples).

    Args:
        X_train (numpy.ndarray): The training data features.
        y_train (numpy.ndarray): The training data labels.
        num_clients (int): The number of clients among which the data should be distributed.
        biased_labels_list (list): The list of labels that should be non-uniformly distributed.

    Returns:
        list of tf.data.Dataset: A list of TensorFlow Dataset objects. Each dataset in the list corresponds to
        the data shard for a client. The order of the datasets in the list corresponds to the order of the clients.
    """

    # Separate the data based on the biased labels
    mask = np.isin(y_train, biased_labels_list).ravel()

    X_train_biased = X_train[mask]
    y_train_biased = y_train[mask]

    X_train_rest = X_train[~mask]
    y_train_rest = y_train[~mask]

    # Concatenate the biased and rest datasets
    X_train_multi_label_biased = np.concatenate((X_train_biased, X_train_rest))
    y_train_multi_label_biased = np.concatenate((y_train_biased, y_train_rest))

    # Split the data into parts for each client
    X_train_multi_label_biased_lst = np.array_split(X_train_multi_label_biased, num_clients)
    y_train_multi_label_biased_lst = np.array_split(y_train_multi_label_biased, num_clients)

    # Create TensorFlow datasets for each client
    multi_label_biased_federated_dataset = []
    for X_train_client, y_train_client in zip(X_train_multi_label_biased_lst, y_train_multi_label_biased_lst):
        dataset = tf.data.Dataset.from_tensor_slices((X_train_client, y_train_client))
        multi_label_biased_federated_dataset.append(dataset)

    return multi_label_biased_federated_dataset

