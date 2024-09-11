from fdavg.metrics.epoch_metrics import EpochMetrics
from fdavg.models.miscellaneous import (average_trainable_client_weights, synchronize_clients, current_accuracy,
                                        average_non_trainable_client_weights)
import tensorflow as tf
import gc


def clients_train_fed_opt(client_cnns, federated_dataset):
    """
    Train all client models with FedOpt

    Args:
    - client_cnns (list): A list of client CNN model objects.
    - federated_dataset (list of tf.data.Dataset): A list of datasets, one for each client.

    Returns:
    None
    """
    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        client_cnn.train(client_dataset)


def fed_opt_simulation(test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs,
                       steps_in_one_epoch):
    """
    Run a federated learning simulation with FedOpt
    This function collects epoch-wise metrics.

    Args:
    - test_dataset (tf.data.Dataset): The dataset for evaluating the model's performance.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for each client model.
    - server_cnn (object): The server's CNN model object.
    - client_cnns (list): A list of client CNN model objects.
    - num_epochs (int): The total number of epochs to run.
    - steps_in_one_epoch (int): The number of steps in one epoch (number of
        steps per `clients_train_fed_opt` calls).
    - E (int): The number of epochs until round ends.

    Returns:
    - epoch_metrics_list (list): A list of tuples containing epoch-wise metrics.

    Note:
    - A step and a round is the same thing in this method.
    - The function also prints metrics at the end of each epoch for monitoring.
    """

    # Initialize various counters and metrics
    tmp_steps = 0  # Counter to monitor when epochs pass
    epoch_count = 0  # Epoch counter
    total_rounds = 1  # Round counter
    total_steps = 0  # Total steps taken

    synchronize_clients(server_cnn, client_cnns)  # Set clients to server model

    # Initialize list for storing epoch metrics
    epoch_metrics_list = []

    while epoch_count <= num_epochs:

        if total_steps % 100 == 0:
            gc.collect()

        # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
        clients_train_fed_opt(client_cnns, federated_dataset)

        # Remove
        # print(f"1: {client_cnns[0].metrics[1].result().numpy()} 2: {client_cnns[0].metrics[0].result().numpy()}")

        tmp_steps += 1
        total_steps += 1

        # If Epoch has passed in this step then round ends
        if tmp_steps >= steps_in_one_epoch:

            # Minus here and not `tmp_steps = 0` because `steps_in_one_epoch` is not an integer necessarily,
            # and we need to keep track of potentially more data seen in this step (many clients, large batch sizes)
            tmp_steps -= steps_in_one_epoch

            avg_trainable_variables = average_trainable_client_weights(client_cnns)

            # List of pseudo-gradients for each layer. Note that they are tf.Tensor and not tf.Variables (as we want).
            pseudo_gradient = [
                layer_w_t0 - avg_layer_w_t
                for layer_w_t0, avg_layer_w_t
                in zip(server_cnn.trainable_variables, avg_trainable_variables)
            ]

            # apply pseudo-gradient on server
            server_cnn.optimizer.apply_gradients(zip(pseudo_gradient, server_cnn.trainable_variables))

            # aggregate non-trainable variables on server because we are about to evaluate
            avg_non_trainable_variables = average_non_trainable_client_weights(client_cnns)
            server_cnn.set_non_trainable_variables(avg_non_trainable_variables)

            synchronize_clients(server_cnn, client_cnns)

            epoch_count += 1
            total_rounds += 1

            # ---------- Metrics ------------
            _, acc = server_cnn.evaluate(test_dataset, verbose=0)
            train_acc = tf.reduce_mean([cnn.metrics[1].result() for cnn in client_cnns]).numpy()
            epoch_metrics = EpochMetrics(epoch_count, total_rounds, total_steps, acc, train_acc)
            epoch_metrics_list.append(epoch_metrics)
            print(epoch_metrics)  # remove
            # -------------------------------

            # Reset training accuracy
            for cnn in client_cnns:
                cnn.metrics[1].reset_state()

    return epoch_metrics_list
