from fdavg.metrics.epoch_metrics import EpochMetrics
from fdavg.models.miscellaneous import average_trainable_client_weights, synchronize_clients, current_accuracy
import tensorflow as tf
import gc

from fdavg.utils.communication_cost import comm_cost_str
from fdavg.models.miscellaneous import count_weights
from fdavg.utils.pretty_printers import print_epoch_metrics

def clients_train_synchronous(client_cnns, federated_dataset):
    """
    Train all client models with the synchronous method (FDA with theta equal to zero).
    
    Args:
    - client_cnns (list): A list of client CNN model objects.
    - federated_dataset (list of tf.data.Dataset): A list of datasets, one for each client.
    
    Returns:
    None
    """
    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        client_cnn.train(client_dataset)


def synchronous_federated_simulation(test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs,
                                     fda_steps_in_one_epoch, compile_and_build_model_func):
    """
    Run a federated learning simulation with the synchronous method (FDA with theta equal to zero).
    This function collects epoch-wise metrics.
    
    Args:
    - test_dataset (tf.data.Dataset): The dataset for evaluating the model's performance.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for each client model.
    - server_cnn (object): The server's CNN model object.
    - client_cnns (list): A list of client CNN model objects.
    - num_epochs (int): The total number of epochs to run.
    - fda_steps_in_one_epoch (int): The number of FDA steps in one epoch (number of
        steps per `clients_train_synchronous` calls).
    - compile_and_build_model_func (callable): Function to compile and build the model.
    
    Returns:
    - epoch_metrics_list (list): A list of tuples containing epoch-wise metrics.
    
    Note:
    - An FDA step and a round is the same thing in this method.
    - The function also prints metrics at the end of each epoch for monitoring.
    """
    
    # Initialize various counters and metrics
    tmp_fda_steps = 0  # Counter to monitor when epochs pass
    epoch_count = 1  # Epoch counter
    total_rounds = 1  # Round counter
    total_fda_steps = 0  # Total FDA steps taken

    nn_num_weights = count_weights(server_cnn)
    num_clients = len(client_cnns)

    synchronize_clients(server_cnn, client_cnns)  # Set clients to server model

    # Initialize list for storing epoch metrics
    epoch_metrics_list = []

    # Temporary model to evaluate the testing accuracy on, without messing up the training process
    tmp_model_for_acc = compile_and_build_model_func()
    
    while epoch_count <= num_epochs:

        if total_fda_steps % 25 == 0:
            gc.collect()
            
        # train clients, each on some number of batches which depends on `.take` creation of dataset (Default=1)
        clients_train_synchronous(client_cnns, federated_dataset)

        tmp_fda_steps += 1
        total_fda_steps += 1

        comm_cost = comm_cost_str(total_fda_steps, total_rounds, num_clients, nn_num_weights, 'synchronous')
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
            print_epoch_metrics(epoch_metrics)
            # print([cnn.optimizer.learning_rate.numpy() for cnn in client_cnns])
            # -------------------------------

            # Reset training accuracy
            for cnn in client_cnns:
                cnn.metrics[1].reset_state()

            epoch_count += 1
        
        # Round finished
        print(f"Synchronizing...!")

        # server average
        server_cnn.set_trainable_variables(average_trainable_client_weights(client_cnns))
        
        synchronize_clients(server_cnn, client_cnns)

        total_rounds += 1

    return epoch_metrics_list

                