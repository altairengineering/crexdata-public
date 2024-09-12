import tensorflow as tf

from fdavg.metrics.epoch_metrics import EpochMetrics
from fdavg.models.miscellaneous import (average_trainable_client_weights, weighted_average_client_weights,
                                        current_accuracy, synchronize_clients)
import gc

from fdavg.utils.communication_cost import comm_cost_str
from fdavg.models.miscellaneous import count_weights
from fdavg.utils.pretty_printers import print_epoch_metrics

def ksi_unit(w_t0, w_tminus1):
    """
    Calculates the heuristic unit vector ksi.

    Args:
    - w_t0 (tf.Tensor): Initial model parameters for the current round. Shape=(d,).
    - w_tminus1 (tf.Tensor): Model parameters from the previous round. Shape=(d,).

    Returns:
    - tf.Tensor: The heuristic unit vector ksi.
    """
    if tf.reduce_all(tf.equal(w_t0, w_tminus1)):
        # if equal then ksi becomes a random vector (will only happen in round 1)
        ksi = tf.random.stateless_normal(shape=w_t0.shape, seed=[1, 2])

    else:
        ksi = w_t0 - w_tminus1

    # Normalize and return
    return tf.divide(ksi, tf.norm(ksi))


def client_train_linear(w_t0, w_tminus1, client_cnn, client_dataset):
    """
    Trains a client model and returns the square of the Euclidean norm of the update vector and the
    dot product with ksi.

    Args:
    - w_t0 (tf.Tensor): Initial model parameters for the current round. Shape=(d,).
    - w_tminus1 (tf.Tensor): Model parameters from the previous round. Shape=(d,).
    - client_cnn (object): The client's CNN model.
    - client_dataset (tf.data.Dataset): The dataset on which the client model will be trained.

    Returns:
    - tuple: (Square of the Euclidean norm of the update vector, Dot product with ksi), both as tf.Tensor with shape=().
    """
    
    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)
    
    Delta_i = client_cnn.trainable_vars_as_vector() - w_t0
    
    # ||D(t)_i||^2 , shape = ()
    Delta_i_euc_norm_squared = tf.reduce_sum(tf.square(Delta_i))  # ||D(t)_i||^2
    
    # heuristic unit vector ksi
    ksi = ksi_unit(w_t0, w_tminus1)
    
    # ksi * Delta_i (* is dot) , shape = ()
    ksi_Delta_i = tf.reduce_sum(tf.multiply(ksi, Delta_i))
    
    return Delta_i_euc_norm_squared, ksi_Delta_i


def clients_train_linear(w_t0, w_tminus1, client_cnns, federated_dataset):
    """
    Trains multiple client models and returns two lists containing the square of the Euclidean
    norms and the dot products with ksi.

    Args:
    - w_t0 (tf.Tensor): Initial model parameters for the current round. Shape=(d,).
    - w_tminus1 (tf.Tensor): Model parameters from the previous round. Shape=(d,).
    - client_cnns (list): A list of client CNN models.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for each client model.

    Returns:
    - tuple: Two lists containing the square of the Euclidean norms and the dot products with ksi for each client.
    """
    
    euc_norm_squared_clients = []
    ksi_delta_clients = []
    
    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        Delta_i_euc_norm_squared, ksi_Delta_i = client_train_linear(
            w_t0, w_tminus1, client_cnn, client_dataset
        )

        euc_norm_squared_clients.append(Delta_i_euc_norm_squared)
        ksi_delta_clients.append(ksi_Delta_i)
    
    return euc_norm_squared_clients, ksi_delta_clients


def f_linear(euc_norm_squared_clients, ksi_delta_clients):
    """
    Calculates the linear approximation of the variance based on the given lists.

    Args:
    - euc_norm_squared_clients (list of tf.Tensor): List of squares of the Euclidean norms for each client.
    - ksi_delta_clients (list of tf.Tensor): List of dot products with ksi for each client.

    Returns:
    - tf.Tensor: Linear variance approximation. Shape=(), dtype=tf.float32.
    """
    
    S_1 = tf.reduce_mean(euc_norm_squared_clients)
    S_2 = tf.reduce_mean(ksi_delta_clients)
    
    return S_1 - S_2**2


def linear_federated_simulation(test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, theta, 
                                fda_steps_in_one_epoch, compile_and_build_model_func, aggr_scheme):
    """
    Run a federated learning simulation using the Linear FDA method. 
    Collects both general and time-series-like metrics.
    
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
    - An FDA step is considered as a single update from each client.
    - The function also prints metrics at the end of each epoch and round for monitoring.
    """
    
    # Initialize counters and metrics lists
    tmp_fda_steps = 0  # Counter for FDA steps within an epoch
    epoch_count = 1  # Current epoch number
    total_rounds = 1  # Total number of rounds completed
    total_fda_steps = 0  # Total number of FDA steps taken
    est_var = 0  # Estimated variance

    nn_num_weights = count_weights(server_cnn)
    num_clients = len(client_cnns)

    synchronize_clients(server_cnn, client_cnns)  # Set clients to server model

    w_t0 = server_cnn.trainable_vars_as_vector()
    w_tminus1 = w_t0   # Initialize w_tminus1 to be the same as w_t0 for the first round
    
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
            euc_norm_squared_clients, ksi_delta_clients = clients_train_linear(
                w_t0, w_tminus1, client_cnns, federated_dataset
            )

            # Linear estimation of variance
            est_var = f_linear(euc_norm_squared_clients, ksi_delta_clients).numpy()
            
            tmp_fda_steps += 1
            total_fda_steps += 1

            comm_cost = comm_cost_str(total_fda_steps, total_rounds, num_clients, nn_num_weights, 'linear')
            print(f"Step {total_fda_steps} , Communication Cost: {comm_cost}")

            #print(f"Step {tmp_fda_steps}/{fda_steps_in_one_epoch} ,  est_var: {est_var:.2f}")
            
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
            # TODO: follow above scheme w/ non-trainable params
            sum_delta_i = tf.reduce_sum(euc_norm_squared_clients)
            weights = [tf.divide(delta_i_sq, sum_delta_i) for delta_i_sq in euc_norm_squared_clients]
            server_cnn.set_trainable_variables(weighted_average_client_weights(client_cnns, weights))
        else:
            print("Unrecognized aggregation scheme")
            return

        w_tminus1 = w_t0
        w_t0 = server_cnn.trainable_vars_as_vector()

        # clients sync
        synchronize_clients(server_cnn, client_cnns)
        est_var = 0

        total_rounds += 1
        
    return epoch_metrics_list



                