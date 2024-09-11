import tensorflow as tf
import numpy as np


def count_weights(model):
    """
    Count the total number of trainable parameters in a Keras model.

    This function iterates through all layers of the provided Keras model and counts
    the total number of trainable parameters (weights and biases).

    Args:
    - model (keras.Model): A Keras model object. It is expected to have an attribute `layers`,
      which is a list of layers in the model.

    Returns:
    - total_params (int): The total number of trainable parameters in the model.

    Example:
    If the model has a dense layer with 128 units and input dimension of 784, and another dense
    layer with 10 units, the total number of parameters would be:
    (128 * 784) + 128 (weights and biases for the first layer) + (128 * 10) + 10 (weights and
    biases for the second layer) = 100480 + 1380 = 101860
    """
    
    total_params = 0
    for layer in model.layers:
        total_params += np.sum([np.prod(weight.shape) for weight in layer.trainable_weights])
    return int(total_params)


def current_accuracy(client_models, test_dataset, tmp_model):
    """
    Compute the current test accuracy with the specific models weights.

    This function takes a list of client models, averages their trainable parameters,
    and then evaluates the test accuracy on a given test dataset using these averaged parameters.

    Args:
    - model_weights (list of tf.Tensor): The current model parameters.
    - test_dataset (tf.data.Dataset): A TensorFlow Dataset object representing the test data.
    - tmp_model (tf.keras.Model): Temporary model to test accuracy with

    Returns:
    - acc (float): The test accuracy of the model with averaged client weights on the provided test dataset.
    """
    avg_trainable_weights = average_trainable_client_weights(client_models)
    avg_non_trainable_weights = average_non_trainable_client_weights(client_models)

    tmp_model.set_trainable_variables(avg_trainable_weights)
    tmp_model.set_non_trainable_variables(avg_non_trainable_weights)

    # Evaluate the temporary model on the test dataset
    _, acc = tmp_model.evaluate(test_dataset, verbose=0)

    return acc


def average_trainable_client_weights(client_models):
    """
    Compute the average of the trainable parameters across multiple client models.

    This function takes a list of client models and calculates the average of their 
    trainable parameters. The averaging is done layer-wise, meaning that the average 
    for each layer is computed separately and then returned as a list of average weights 
    for each layer.

    https://stackoverflow.com/questions/48212110/average-weights-in-keras-models

    Args:
    - client_models (list of objects): A list of objects representing the client models. 
      Each object is expected to have an attribute `trainable_variables` that returns a 
      list of `tf.Variable` objects representing the trainable parameters of the model.

    Returns:
    - avg_weights (list of tf.Tensor): A list of tensors representing the average weights 
      of the trainable parameters of the client models. Each tensor in the list corresponds 
      to the average weight for a specific layer.

    Example:
    If client_models[0].trainable_variables = [W1, b1, W2, b2], where W1, b1, W2, b2 are 
    tensors, then avg_weights = [avg_W1, avg_b1, avg_W2, avg_b2], where avg_W1, avg_b1, 
    avg_W2, avg_b2 are the average weights for each corresponding layer.
    """
    # Retrieve the trainable variables from each client model
    client_weights = [model.trainable_variables for model in client_models]

    # Compute the average weights for each layer
    avg_weights = [
        tf.reduce_mean(layer_weight_tensors, axis=0)
        for layer_weight_tensors in zip(*client_weights)
    ]

    return avg_weights


def average_non_trainable_client_weights(client_models):
    """
    Compute the average of the non-trainable parameters across multiple client models.

    This function takes a list of client models and calculates the average of their
    non-trainable parameters. The averaging is done layer-wise, meaning that the average
    for each layer is computed separately and then returned as a list of average weights
    for each layer.

    https://stackoverflow.com/questions/48212110/average-weights-in-keras-models

    Args:
    - client_models (list of objects): A list of objects representing the client models.
      Each object is expected to have an attribute `non_trainable_variables` that returns a
      list of `tf.Variable` objects representing the non-trainable parameters of the model.

    Returns:
    - avg_weights (list of tf.Tensor): A list of tensors representing the average weights
      of the non-trainable parameters of the client models. Each tensor in the list corresponds
      to the average weight for a specific layer.
    """
    # Retrieve the trainable variables from each client model
    client_non_trainable_weights = [model.non_trainable_variables for model in client_models]

    # Compute the average weights for each layer
    avg_non_trainable_weights = [
        tf.reduce_mean(layer_weight_tensors, axis=0)
        for layer_weight_tensors in zip(*client_non_trainable_weights)
    ]

    return avg_non_trainable_weights


def weighted_average_client_weights(client_models, weights):
    """
    Compute the weighted average of the trainable parameters across multiple client models.

    This function takes a list of client models and their associated weights, and calculates the weighted
    average of their trainable parameters. The averaging is done layer-wise, meaning that the weighted
    average for each layer is computed separately and then returned as a list of weighted average weights
    for each layer.

    Args:
    - client_models (list of objects): A list of objects representing the client models. Each object
      is expected to have an attribute `trainable_variables` that returns a list of `tf.Variable` objects
      representing the trainable parameters of the model.
    - weights (list of tf.Tensor floats, i.e., shape=()): A list of weights, each corresponding to a client model in
      `client_models`. These weights are used to calculate the weighted average of the model parameters. The length
      of this list should match the number of client models.

    Returns:
    - weighted_avg_weights (list of tf.Tensor): A list of tensors representing the weighted average weights
      of the trainable parameters of the client models. Each tensor in the list corresponds to the weighted
      average weight for a specific layer.

    Example:
    If client_models[0].trainable_variables = [W1, b1, W2, b2], where W1, b1, W2, b2 are tensors, and
    weights = [0.3, 0.7], then weighted_avg_weights = [weighted_avg_W1, weighted_avg_b1, weighted_avg_W2,
    weighted_avg_b2], where weighted_avg_W1, weighted_avg_b1, weighted_avg_W2, weighted_avg_b2 are the
    weighted average weights for each corresponding layer, with the weights applied accordingly.
    """
    def scale_layer_weights(_layer_weight_tensors, _weights):
        # Scale each tensor by its corresponding weight
        scaled_weights = [tensor * weight for tensor, weight in zip(_layer_weight_tensors, _weights)]

        return scaled_weights

    # Retrieve the trainable variables from each client model
    client_weights = [model.trainable_variables for model in client_models]

    # Compute the average weights for each layer
    weighted_avg_weights = [
        tf.reduce_sum(scale_layer_weights(layer_weight_tensors, weights), axis=0)
        for layer_weight_tensors in zip(*client_weights)
    ]

    return weighted_avg_weights


def avg_client_layer_weights(layer_i, client_models):
    # Here, each client_layer_weights[i] corresponds to a list of tf.Variables for the `layer_i` of the i-th client
    # Note: Most layers have the actual weights and the biases, so usually each such client list is len 2
    client_layer_weights = [model.layers[layer_i].trainable_weights for model in client_models]

    # Stack all client weights horizontally, i.e., all actual weights together, all biases together and average.
    avg_layer_weights = [
        tf.reduce_mean(layer_vars, axis=0)
        for layer_vars in zip(*client_layer_weights)
    ]

    return avg_layer_weights


def synchronize_clients(server_model, client_models):
    """
    Synchronize the trainable parameters of client models with those of a server model.

    This function takes a server model and a list of client models, and sets the trainable
    parameters of each client model to be equal to those of the server model. This effectively
    synchronizes the client models with the server model.

    Args:
    - server_model (object): An object representing the server model. It is expected to have an
      attribute `trainable_variables` that returns a list of `tf.Variable` objects representing
      the trainable parameters of the model.
    - client_models (list of objects): A list of objects representing the client models. Each object
      is expected to have a method `set_trainable_variables` that allows setting the trainable
      variables of the model.
    """
    for client_model in client_models:
        client_model.set_trainable_variables(server_model.trainable_variables)


def set_trainable_variables(model, trainable_vars):
    """
    Set the model's trainable variables.

    Args:
    - model (object): An object representing the model. It is expected to have an attribute `trainable_variables`
    - trainable_vars (list of tf.Tensor): A list of tensors representing the trainable variables to be set.

    This method sets each of the model's trainable variables to the corresponding tensor in `trainable_vars`.
    """
    for model_var, var in zip(model.trainable_variables, trainable_vars):
        model_var.assign(var)


def trainable_vars_as_vector(model):
    """
    Get the model's trainable variables as a single vector.

    Returns:
    - tf.Tensor: A 1D tensor containing all of the model's trainable variables.
    """
    return tf.concat([tf.reshape(var, [-1]) for var in model.trainable_variables], axis=0)
