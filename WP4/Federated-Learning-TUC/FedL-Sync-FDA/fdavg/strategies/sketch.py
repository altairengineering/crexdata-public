import tensorflow as tf
import numpy as np

from fdavg.metrics.epoch_metrics import EpochMetrics
from fdavg.models.miscellaneous import (average_trainable_client_weights, weighted_average_client_weights,
                                        current_accuracy, synchronize_clients)

import gc

from fdavg.utils.communication_cost import comm_cost_str
from fdavg.models.miscellaneous import count_weights
from fdavg.utils.pretty_printers import print_epoch_metrics

class AmsSketch:
    """
    AMS Sketch class for approximate second moment estimation.
    """

    def __init__(self, depth=5, width=250, with_seed=False, use_other_gpu=None, save_on_cpu=False, chunk_size=None):
        self.save_on_cpu = save_on_cpu
        self.chunk_size = chunk_size
        self.use_other_gpu = use_other_gpu

        self.depth = tf.constant(depth)
        self.width = tf.constant(width)

        if with_seed:
            self.F = tf.random.stateless_uniform(shape=(6, depth), minval=0, maxval=(1 << 31) - 1, dtype=tf.int32,
                                                 seed=(1, 2))
        else:
            self.F = tf.random.uniform(shape=(6, depth), minval=0, maxval=(1 << 31) - 1, dtype=tf.int32)

        self.zeros_sketch = tf.zeros(shape=(self.depth, self.width), dtype=tf.float32)

        self.precomputed_dict = {}

    def precompute(self, d):
        pos_tensor = self.tensor_hash31(tf.range(d), self.F[0], self.F[1]) % self.width  # shape=(d, 5)

        self.precomputed_dict[('four', d)] = tf.cast(self.tensor_fourwise(tf.range(d)),
                                                     dtype=tf.float32)  # shape=(d, 5)

        range_tensor = tf.range(self.depth)  # shape=(5,)

        # Expand dimensions to create a 2D tensor with shape (1, `self.depth`)
        range_tensor_expanded = tf.expand_dims(range_tensor, 0)  # shape=(1, 5)

        # Use tf.tile to repeat the range `d` times
        repeated_range_tensor = tf.tile(range_tensor_expanded, [d, 1])  # shape=(d, 5)

        # shape=(`d`, `self.depth`, 2)
        self.precomputed_dict[('indices', d)] = tf.stack([repeated_range_tensor, pos_tensor],
                                                         axis=-1)  # shape=(d, 5, 2)

    @staticmethod
    def hash31(x, a, b):
        r = a * x + b
        fold = tf.bitwise.bitwise_xor(tf.bitwise.right_shift(r, 31), r)
        return tf.bitwise.bitwise_and(fold, 2147483647)

    @staticmethod
    def tensor_hash31(x, a, b):  # GOOD
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., tf.range(d)) """

        # Reshape x to have an extra dimension, resulting in a shape of (k, 1)
        x_reshaped = tf.expand_dims(x, axis=-1)

        # shape=(`v_dim`, 7)
        r = tf.multiply(a, x_reshaped) + b

        fold = tf.bitwise.bitwise_xor(tf.bitwise.right_shift(r, 31), r)

        return tf.bitwise.bitwise_and(fold, 2147483647)

    def tensor_fourwise(self, x):
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., tf.range(d)) """

        # 1st use the tensor hash31
        in1 = self.tensor_hash31(x, self.F[2], self.F[3])  # shape = (`x_dim`,  `self.depth`)

        # 2st use the tensor hash31
        in2 = self.tensor_hash31(x, in1, self.F[4])  # shape = (`x_dim`,  `self.depth`)

        # 3rd use the tensor hash31
        in3 = self.tensor_hash31(x, in2, self.F[5])  # shape = (`x_dim`,  `self.depth`)

        in4 = tf.bitwise.bitwise_and(in3, 32768)  # shape = (`x_dim`,  `self.depth`)

        return 2 * (tf.bitwise.right_shift(in4, 15)) - 1  # shape = (`x_dim`,  `self.depth`)

    def fourwise(self, x):
        result = 2 * (tf.bitwise.right_shift(tf.bitwise.bitwise_and(
            self.hash31(self.hash31(self.hash31(x, self.F[2], self.F[3]), x, self.F[4]), x, self.F[5]), 32768), 15)) - 1
        return result

    def sketch_for_vector(self, v):
        """ Extremely efficient computation of sketch with only using tensors.

        Args:
        - v (tf.Tensor): Vector to sketch. Shape=(d,).

        Returns:
        - tf.Tensor: An AMS - Sketch. Shape=(`depth`, `width`).
        """

        d = v.shape[0]

        if ('four', d) not in self.precomputed_dict:
            if self.save_on_cpu:
                with tf.device('/CPU:0'):
                    self.precompute(d)
            else:
                self.precompute(d)

        four, indices = self.precomputed_dict[('four', d)], self.precomputed_dict[('indices', d)]

        if self.use_other_gpu:

            with tf.device(self.use_other_gpu):

                if self.chunk_size:
                    return self._sketch_for_vector_with_chunks(v, four, indices)

                return self._sketch_for_vector(v, four, indices)

        if self.chunk_size:
            return self._sketch_for_vector_with_chunks(v, four, indices)

        return self._sketch_for_vector(v, four, indices)

    @tf.function
    def _sketch_for_vector(self, v, four, indices):
        v_expand = tf.expand_dims(v, axis=-1)  # shape=(d, 1)

        # shape=(d, 5): +- for each value v_i , i = 1, ..., d
        deltas_tensor = tf.multiply(four, v_expand)

        sketch = tf.tensor_scatter_nd_add(self.zeros_sketch, indices, deltas_tensor)  # shape=(5, 250)

        return sketch

    def _sketch_for_vector_with_chunks(self, v, four, indices):
        """ Less memory intensive version of `_sketch_for_vector`, working with chunks of numbers"""

        sketch = tf.zeros(shape=(self.depth, self.width), dtype=tf.float32)

        d = v.shape[0]

        for start in range(0, d, self.chunk_size):
            end = min(start + self.chunk_size, d)

            # Slice the tensors into chunks
            four_chunk = four[start:end, :]
            v_chunk = v[start:end]
            indices_chunk = indices[start:end, :, :]

            sketch_from_chunk = self._sketch_for_vector(v_chunk, four_chunk, indices_chunk)

            sketch = tf.add(sketch, sketch_from_chunk)

        return sketch

    @staticmethod
    def estimate_euc_norm_squared(sketch):
        """ Estimate the Euclidean norm squared of a vector using its AMS sketch.

        Args:
        - sketch (tf.Tensor): AMS sketch of a vector. Shape=(`depth`, `width`).

        Returns:
        - tf.Tensor: Estimated squared Euclidean norm.
        """

        norm_sq_rows = tf.reduce_sum(tf.square(sketch), axis=1)
        return np.median(norm_sq_rows)


def client_train_sketch2(w_t0, client_cnn, client_dataset, ams_sketch):   # TODO: Remove fucn
    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)

    Delta_i = client_cnn.trainable_vars_as_vector() - w_t0

    #  ||D(t)_i||^2 , shape = ()
    Delta_i_euc_norm_squared = tf.reduce_sum(tf.square(Delta_i))  # ||D(t)_i||^2

    return Delta_i_euc_norm_squared, Delta_i


def clients_train_sketch2(w_t0, client_cnns, federated_dataset, ams_sketch):  # TODO: Remove fucn
    euc_norm_squared_clients = []

    d = w_t0.shape[0]
    num_clients = len(client_cnns)

    sum_Delta_i = tf.zeros((d,))

    # client steps (number depends on `federated_dataset`, i.e., `.take(num)`)
    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        Delta_i_euc_norm_squared, Delta_i = client_train_sketch2(
            w_t0, client_cnn, client_dataset, ams_sketch
        )

        sum_Delta_i += Delta_i

        euc_norm_squared_clients.append(Delta_i_euc_norm_squared)

    return euc_norm_squared_clients, sum_Delta_i / num_clients


def f_sketch2(euc_norm_squared_clients, mean_Delta_i, epsilon):  # TODO: Remove fucn
    S_1 = tf.reduce_mean(euc_norm_squared_clients)
    S_2 = tf.reduce_sum(tf.square(mean_Delta_i))

    # See theoretical analysis above
    return S_1 - S_2


def client_train_sketch(w_t0, client_cnn, client_dataset, ams_sketch):
    """
    Train a client model and return the AMS sketch and square of the Euclidean norm.

    Args:
    - w_t0 (tf.Tensor): Initial model parameters. Shape=(d,).
    - client_cnn (object): Client CNN model.
    - client_dataset (tf.data.Dataset): Client dataset.
    - ams_sketch (AmsSketch): AMS sketch instance.

    Returns:
    - tuple: (Square of the Euclidean norm, AMS sketch)
    """
    # number of steps depend on `.take()` from `dataset`
    client_cnn.train(client_dataset)

    Delta_i = client_cnn.trainable_vars_as_vector() - w_t0

    #  ||D(t)_i||^2 , shape = ()
    Delta_i_euc_norm_squared = tf.reduce_sum(tf.square(Delta_i))  # ||D(t)_i||^2

    # sketch approx
    sketch = ams_sketch.sketch_for_vector(Delta_i)

    return Delta_i_euc_norm_squared, sketch


def clients_train_sketch(w_t0, client_cnns, federated_dataset, ams_sketch):
    """
    Train multiple client models and return lists of the AMS sketches and the square of the Euclidean norms.
    
    Args:
    - w_t0 (tf.Tensor): Initial model parameters. Shape=(d,).
    - client_cnns (list): List of client CNN models.
    - federated_dataset (list): List of client datasets.
    - ams_sketch (AmsSketch): AMS sketch instance.
    
    Returns:
    - tuple: (List of squares of the Euclidean norms, List of AMS sketches)
    """
    
    euc_norm_squared_clients = []
    sketch_clients = []

    # client steps (number depends on `federated_dataset`, i.e., `.take(num)`)
    for client_cnn, client_dataset in zip(client_cnns, federated_dataset):
        Delta_i_euc_norm_squared, sketch = client_train_sketch(
            w_t0, client_cnn, client_dataset, ams_sketch
        )

        euc_norm_squared_clients.append(Delta_i_euc_norm_squared)
        sketch_clients.append(sketch)
        
    return euc_norm_squared_clients, sketch_clients


def f_sketch(euc_norm_squared_clients, sketch_clients, epsilon):
    """
    Compute the approximation of the variance using sketches.
    
    Args:
    - euc_norm_squared_clients (list): List of squared Euclidean norms.
    - sketch_clients (list): List of AMS sketches.
    - epsilon (float): Error bound for the AMS sketch.
    
    Returns:
    - tf.Tensor: Approximation of the variance.
    """
    S_1 = tf.reduce_mean(euc_norm_squared_clients)
    S_2 = tf.reduce_mean(sketch_clients, axis=0)  # shape=(`depth`, width`). See `Ξ` in theoretical analysis
    
    # See theoretical analysis above
    return S_1 - (1. / (1. + epsilon)) * AmsSketch.estimate_euc_norm_squared(S_2)


def sketch_federated_simulation(test_dataset, federated_dataset, server_cnn, client_cnns, num_epochs, theta, 
                                fda_steps_in_one_epoch, compile_and_build_model_func, ams_sketch, epsilon, aggr_scheme):
    """
    Run a federated learning simulation using the AMS Sketch FDA method.
    This function collects both general and time-series-like metrics.
    
    Args:
    - test_dataset (tf.data.Dataset): The dataset for evaluating the model's performance.
    - federated_dataset (list of tf.data.Dataset): A list of datasets for each client model.
    - server_cnn (object): The server's CNN model object.
    - client_cnns (list): A list of client CNN model objects.
    - num_epochs (int): The total number of epochs to run.
    - theta (float): The threshold for variance.
    - fda_steps_in_one_epoch (int): The number of FDA steps in one epoch.
    - compile_and_build_model_func (callable): Function to compile and build the model.
    - ams_sketch (AmsSketch): An instance of the AMS Sketch class for variance approximation.
    - epsilon (float): Error tolerance for AMS sketch.
    
    Returns:
    - epoch_metrics_list (list): A list of tuples containing epoch-wise metrics.
    
    Note:
    - An FDA step is considered as a single update from each client.
    - The function also prints metrics at the end of each epoch and round for monitoring.
    """
    
    # Initialize various counters and metrics
    tmp_fda_steps = 0  # Counter to monitor when epochs pass
    epoch_count = 1  # Epoch counter
    total_rounds = 1  # Round counter
    total_fda_steps = 0  # Total FDA steps taken
    est_var = 0  # Estimated variance

    nn_num_weights = count_weights(server_cnn)
    num_clients = len(client_cnns)

    synchronize_clients(server_cnn, client_cnns)
    
    # Initialize models and synchronize client and server models
    w_t0 = server_cnn.trainable_vars_as_vector()
    
    # Initialize list for storing metrics
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
            euc_norm_squared_clients, sketch_clients = clients_train_sketch(
                w_t0, client_cnns, federated_dataset, ams_sketch
            )

            # Sketch estimation of variance
            est_var = f_sketch(euc_norm_squared_clients, sketch_clients, epsilon).numpy()

            tmp_fda_steps += 1
            total_fda_steps += 1

            comm_cost = comm_cost_str(total_fda_steps, total_rounds, num_clients, nn_num_weights, 'sketch')
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
                