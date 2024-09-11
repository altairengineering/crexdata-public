import tensorflow as tf
import copy

from fdavg.metrics.epoch_metrics import EpochMetrics
from fdavg.models.miscellaneous import (average_trainable_client_weights, weighted_average_client_weights,
                                        current_accuracy, synchronize_clients, avg_client_layer_weights)

import gc

from fdavg.utils.communication_cost import comm_cost_str
from fdavg.models.miscellaneous import count_weights
from fdavg.utils.pretty_printers import print_epoch_metrics

def client_train_naive(w_t0, client_cnn, client_dataset):
    """
    Trains a single client model and returns the square of the Euclidean norm of the update vector.

    Args:
    - w_t0 (tf.Tensor): Initial model parameters as a vector. Shape=(d,).
    - client_cnn (object): The client's CNN model.
    - client_dataset (tf.data.Dataset): The dataset on which the client model will be trained.

    Returns:
    - tf.Tensor: The square of the Euclidean norm of the update vector. Shape=(), dtype=tf.float32.
    """
    
    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)
    
    Delta_i = client_cnn.trainable_vars_as_vector() - w_t0
    
    #  ||D(t)_i||^2 , shape = ()
    Delta_i_euc_norm_squared = tf.reduce_sum(tf.square(Delta_i))  # ||D(t)_i||^2
    
    return Delta_i_euc_norm_squared


def clients_train_naive(w_t0, client_cnns, federated_dataset):
    """
    Trains multiple client models and returns the squares of the Euclidean norms of their update vectors.

    Args:
    - w_t0 (tf.Tensor): Initial model parameters as a vector. Shape=(d,).
    - client_cnns (list): A list of client CNN models.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for the client models.

    Returns:
    - list: A list of tensors, each representing the square of the Euclidean norm of the update vector for each client.
    """
    
    S_i_clients = []

    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        Delta_i_euc_norm_squared = client_train_naive(w_t0, client_cnn, client_dataset)
        S_i_clients.append(Delta_i_euc_norm_squared)
    
    return S_i_clients


def f_naive(S_i_clients):
    """
    Calculates the naive approximation of the variance based on the squares of the Euclidean
    norms of the update vectors.

    Args:
    - S_i_clients (list of tf.Tensor): A list of tensors, each representing the square of the
        Euclidean norm of the update vector for each client.

    Returns:
    - tf.Tensor: Naive variance approximation. Shape=(), dtype=tf.float32.
    """

    S = tf.reduce_mean(S_i_clients)

    return S


def naive_federated_simulation(test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, theta, 
                               fda_steps_in_one_epoch, compile_and_build_model_func, aggr_scheme):
    """
    Run a federated learning simulation using the Naive FDA method and collect general and time-series metrics.

    Args:
    - test_dataset (tf.data.Dataset): The dataset for evaluating the model's performance.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for the client models.
    - server_cnn (object): The server's CNN model.
    - client_cnns (list): A list of client CNN models.
    - num_epochs (int): The number of epochs to run the simulation for.
    - theta (float): The threshold for the variance of the update vectors.
    - fda_steps_in_one_epoch (int): The number of FDA steps in one epoch.
    - compile_and_build_model_func (function): Function to compile and build the model.

    Returns:
    - epoch_metrics_list (list): A list of EpochMetrics namedtuples, storing metrics per epoch.

    Note:
    - We consider an FDA step to be a single update from each client.
    - The function also outputs the metrics at the end of each epoch and round for monitoring.
    """
    
    # Initialize counters and metrics lists
    tmp_fda_steps = 0  # Counter for FDA steps within an epoch
    epoch_count = 1  # Current epoch number
    total_rounds = 1  # Total number of rounds completed
    total_fda_steps = 0  # Total number of FDA steps taken
    est_var = 0  # Estimated variance

    nn_num_weights = count_weights(server_cnn)
    num_clients = len(client_cnns)

    synchronize_clients(server_cnn, client_cnns)
    
    # Initialize models and weights
    w_t0 = server_cnn.trainable_vars_as_vector()
    
    # Initialize lists for storing metrics
    epoch_metrics_list = []

    euc_norm_squared_clients = None

    # Temporary model to evaluate the testing accuracy on, without messing up the training process
    tmp_model_for_acc = compile_and_build_model_func()
    
    while epoch_count <= num_epochs:
        
        # Continue training until estimated variance crosses the threshold
        while est_var <= theta:

            if total_fda_steps % 25 == 0:
                gc.collect()

            # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
            euc_norm_squared_clients = clients_train_naive(w_t0, client_cnns, federated_dataset)

            # Naive estimation of variance
            est_var = f_naive(euc_norm_squared_clients).numpy()

            tmp_fda_steps += 1
            total_fda_steps += 1

            comm_cost = comm_cost_str(total_fda_steps, total_rounds, num_clients, nn_num_weights, 'naive')
            print(f"Step {total_fda_steps} , Communication Cost: {comm_cost}")
            
            # If Epoch has passed in this fda step
            if tmp_fda_steps >= fda_steps_in_one_epoch:
                
                # Minus here and not `tmp_fda_steps = 0` because `fda_steps_in_one_epoch` is not an integer necessarily
                # and we need to keep track of potentially more data seen in this fda step
                # (many clients, large batch sizes)
                tmp_fda_steps -= fda_steps_in_one_epoch
                
                # ---------- Metrics ------------
                acc = current_accuracy(client_cnns, test_dataset, tmp_model_for_acc)
                train_acc = tf.reduce_mean([cnn.metrics[1].result() for cnn in client_cnns]).numpy()
                epoch_metrics = EpochMetrics(epoch_count, total_rounds, total_fda_steps, acc, train_acc)
                epoch_metrics_list.append(epoch_metrics)
                print(epoch_metrics)  # remove
                # -------------------------------

                # Reset training accuracy
                for cnn in client_cnns:
                    cnn.metrics[1].reset_state()
                
                epoch_count += 1
                
                if epoch_count > num_epochs:
                    break
        
        # Round finished
        print(f"Synchronizing...!")

        # aggregation
        if aggr_scheme == 'avg':
            server_cnn.set_trainable_variables(average_trainable_client_weights(client_cnns))
        elif aggr_scheme == 'wavg_drifts':
            sum_delta_i = tf.reduce_sum(euc_norm_squared_clients)
            weights = [tf.divide(delta_i_sq, sum_delta_i) for delta_i_sq in euc_norm_squared_clients]
            server_cnn.set_trainable_variables(weighted_average_client_weights(client_cnns, weights))
        else:
            print("Unrecognized aggregation scheme")
            return

        w_t0 = server_cnn.trainable_vars_as_vector()

        # clients sync
        synchronize_clients(server_cnn, client_cnns)
        est_var = 0

        total_rounds += 1

    return epoch_metrics_list


def client_train_naive_per_layer(w_t0, client_cnn, client_dataset):
    """
    Trains a single client model and returns the square of the Euclidean norm of the update vector.

    Args:
    - w_t0 (list): List of the initial per-layer model parameters as a vector.
    - client_cnn (object): The client's CNN model.
    - client_dataset (tf.data.Dataset): The dataset on which the client model will be trained.

    Returns:
    - list: The square of the Euclidean norm of the update vector per-layer. Each one with Shape=(), dtype=tf.float32.
    """

    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)

    deltas = [curr_w - w for curr_w, w in zip(client_cnn.per_layer_trainable_vars_as_vector(), w_t0)]

    # list of ||D(t)_i||^2 , shape = ()
    deltas_squared = [tf.reduce_sum(tf.square(v)) for v in deltas]  # ||D(t)_i||^2

    return deltas_squared


def clients_train_naive_per_layer(w_t0, client_cnns, federated_dataset):
    """
    Trains multiple client models and returns the squares of the Euclidean norms of their update vectors.

    Args:
    - w_t0 (list): List of the initial per-layer model parameters as a vector.
    - client_cnns (list): A list of client CNN models.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for the client models.

    Returns:
    - list(list): A list of a list of tensors. Each i-th inner list represents the square of the Euclidean norm of the
        drift vector for each layer of the i-th client.
    """

    S_i_clients = []

    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        Delta_i_euc_norm_squared = client_train_naive_per_layer(w_t0, client_cnn, client_dataset)
        S_i_clients.append(Delta_i_euc_norm_squared)

    return S_i_clients


def f_naive_per_layer(S_i_clients):
    """
    Calculates the naive approximation of the variance based on the squares of the Euclidean
    norms of the update vectors.

    Args:
    - S_i_clients list(list): A list of a list of tensors. Each i-th inner list represents the square of the Euclidean
        norm of the drift vector for each layer of the i-th client.

    Returns:
    - list(tf.Tensor): Per-layer naive variance approximation. Shape=(), dtype=tf.float32.
    """

    S = [
        tf.reduce_mean(s_per_payer) for s_per_payer in zip(*S_i_clients)
    ]

    return S


def naive_federated_simulation_per_layer(test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, thetas,
                                         fda_steps_in_one_epoch, compile_and_build_model_func, aggr_scheme,
                                         trainable_layers_indices):
    """
    Run a federated learning simulation using the Naive FDA method and collect general and time-series metrics.

    Args:
    - test_dataset (tf.data.Dataset): The dataset for evaluating the model's performance.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for the client models.
    - server_cnn (object): The server's CNN model.
    - client_cnns (list): A list of client CNN models.
    - num_epochs (int): The number of epochs to run the simulation for.
    - thetas (list): A list of floats, thresholds for each layer
    - fda_steps_in_one_epoch (int): The number of FDA steps in one epoch.
    - compile_and_build_model_func (function): Function to compile and build the model.
    - aggr_scheme (string): 'avg' only supported rn
    - trainable_layers_indices (list): List of integers, where each corresponds to the index of a trainable layer.

    Returns:
    - epoch_metrics_list (list): A list of EpochMetrics namedtuples, storing metrics per epoch.

    Note:
    - We consider an FDA step to be a single update from each client.
    - The function also outputs the metrics at the end of each epoch and round for monitoring.
    """

    num_layers = len(trainable_layers_indices)

    # Initialize counters and metrics lists
    tmp_fda_steps = 0  # Counter for FDA steps within an epoch
    epoch_count = 1  # Current epoch number
    total_rounds = [1] * num_layers  # Total number of rounds completed
    total_fda_steps = 0  # Total number of FDA steps taken
    est_var = [0] * num_layers  # Estimated variance

    synchronize_clients(server_cnn, client_cnns)

    # List of per-layer vectors
    w_t0 = server_cnn.per_layer_trainable_vars_as_vector()

    # Initialize lists for storing metrics
    epoch_metrics_list = []

    # euc_norm_squared_clients = None

    # Temporary model to evaluate the testing accuracy on, without messing up the training process
    tmp_model_for_acc = compile_and_build_model_func()

    while epoch_count <= num_epochs:

        # Continue training until any per-layer estimated variance crosses the threshold
        while all([est <= theta for est, theta in zip(est_var, thetas)]):
            # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
            euc_norm_squared_clients = clients_train_naive_per_layer(w_t0, client_cnns, federated_dataset)

            # Naive estimation of variance
            est_var = [est.numpy() for est in f_naive_per_layer(euc_norm_squared_clients)]

            # ------------ FOR TESTING ------------------------------------------------------
            """
            print()
            print(f"Rounds: {total_rounds}")
            print(f"thetas: {thetas}")
            print(f"est_var: {est_var}")
            client_layer_vecs = [
                [w_curr - w_old for w_curr, w_old in zip(client_cnn.per_layer_trainable_vars_as_vector(), w_t0)]
                for client_cnn in client_cnns
            ]
            avg_weights_vecs = [
                tf.reduce_mean(vecs, axis=0) for vecs in zip(*client_layer_vecs)
            ]
            avg_sq_weights = [tf.reduce_sum(tf.square(v)).numpy() for v in avg_weights_vecs]
            actual_var = [d - dm for d, dm in zip(est_var, avg_sq_weights)]
            print(f"act_var: {actual_var}")
            print()
            """
            # ------------ FOR TESTING ------------------------------------------------------

            tmp_fda_steps += 1
            total_fda_steps += 1

            # If Epoch has passed in this fda step
            if tmp_fda_steps >= fda_steps_in_one_epoch:

                # Minus here and not `tmp_fda_steps = 0` because `fda_steps_in_one_epoch` is not an integer necessarily
                # and we need to keep track of potentially more data seen in this fda step
                # (many clients, large batch sizes)
                tmp_fda_steps -= fda_steps_in_one_epoch

                # ---------- Metrics ------------
                acc = current_accuracy(client_cnns, test_dataset, tmp_model_for_acc)
                train_acc = tf.reduce_mean([cnn.metrics[1].result() for cnn in client_cnns]).numpy()
                epoch_metrics = EpochMetrics(epoch_count, copy.copy(total_rounds), total_fda_steps, acc, train_acc)
                epoch_metrics_list.append(epoch_metrics)
                print_epoch_metrics(epoch_metrics)
                # -------------------------------

                # Reset training accuracy
                for cnn in client_cnns:
                    cnn.metrics[1].reset_state()

                epoch_count += 1

                if epoch_count > num_epochs:
                    break

        # Round finished (for some layers)

        # For the layers that the est. of the variance is not bellow the thresh, we aggregate (avg.) them and also
        # store the index `i` that corresponds to the index of each layer in our per-layer lists, and the index
        # `layer_i` which corresponds to the layer of the model that will be used when we update our models (to know
        # which layer corresponds to the avg. weights).
        if aggr_scheme == 'avg':
            layer_index_avg_weights = [
                (i, layer_i, avg_client_layer_weights(layer_i, client_cnns)) for i, (layer_i, bellow_thresh) in
                enumerate(zip(trainable_layers_indices, [est <= theta for est, theta in zip(est_var, thetas)]))
                if not bellow_thresh
            ]
        else:
            print("Unrecognized aggregation scheme")
            return

        # Server - Clients Update (only needed layers)
        # Some layers need synchronization, i.e., the layers that are in `layer_index_avg_weights`. We sync the
        # server model and the client models (only the respective layers of course)
        # For the layers that did change, we need to update the estimated variance of them to 0 and also store that
        # a per-layer round finished (sync happened) for these layers.
        for i, layer_i, layer_weights in layer_index_avg_weights:
            server_cnn.set_layer_weights(layer_i, layer_weights)

            for client_cnn in client_cnns:
                client_cnn.set_layer_weights(layer_i, layer_weights)

            est_var[i] = 0
            total_rounds[i] += 1

        # The old `w_t0` is given by the server's per-layer trainable variables. The layers that did not change are of
        # course the same with the previous `w_t0`. Only the layers that did change will change in the new `w_t0`.
        w_t0 = server_cnn.per_layer_trainable_vars_as_vector()

    return epoch_metrics_list
