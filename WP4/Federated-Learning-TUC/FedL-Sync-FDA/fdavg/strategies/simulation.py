from math import sqrt

from fdavg.metrics.epoch_metrics import TestId, process_metrics_with_test_id
from fdavg.models.miscellaneous import count_weights
from fdavg.strategies.naive import naive_federated_simulation, naive_federated_simulation_per_layer
from fdavg.strategies.linear import linear_federated_simulation
from fdavg.strategies.sketch import sketch_federated_simulation, AmsSketch
from fdavg.strategies.synchronous import synchronous_federated_simulation
from fdavg.strategies.fed_opt import fed_opt_simulation
from fdavg.strategies.gm import gm_federated_simulation


def single_simulation(ds_name, load_federated_data_fn, n_train, fda_name, num_clients, batch_size,
                      num_steps_until_rtc_check, num_epochs, server_compile_and_build_model_fn,
                      client_compile_and_build_model_fn, nn_name, theta=0., bias=None, seed=None, bench_test=False,
                      aggr_scheme='avg', per_layer=False, **kwargs):
    """
    Run a single federated learning simulation based on the given FDA method name.
    
    Args:
        ds_name (str): Name of the dataset.
        load_federated_data_fn (callable): Function to load and preprocess the federated dataset.
        n_train (int): Number of samples in the training dataset.
        fda_name (str): Name of the FDA method to use ("naive", "linear", "sketch", "synchronous").
        num_clients (int): Number of clients participating in the federated learning.
        batch_size (int): The batch size for each client.
        num_steps_until_rtc_check (int): Number of steps to perform before each RTC check.
        num_epochs (int): Number of epochs to run the simulation.
        compile_and_build_model_fn (callable): Function to compile and build the model.
        nn_name (str): Name of the neural network model.
        theta (float, optional): Variance threshold for FDA methods. Defaults to 0. for "synchronous".
        bias (float, optional): Bias parameter for the Fed dataset. Defaults to None.
        seed (int, optional): Random seed the shuffling of the Fed dataset. Defaults to None.
        bench_test (bool, optional): Whether the function is being used for a benchmark test. Defaults to False.
        aggr_scheme (str): either averaging 'avg', or weighted average with drifts 'wavg_drifts'
        per_layer (bool): If True, we train with per-layer theta, else we train with a single theta for all trainable
            variables of the model.

    Returns:
        tuple: A tuple containing two lists:
            - epoch_metrics_with_test_id_list (list): List of epoch metrics with test ID.

    Notes:
        - The function prepares the simulation environment and then runs the simulation based on the `fda_name`.
        - Metrics collected during the simulation are returned with test IDs for identification.
    """

    # 1. Helper variable to count Epochs
    if bench_test:
        fda_steps_in_one_epoch = 10
        steps_in_one_epoch = 10
    else:
        fda_steps_in_one_epoch = ((n_train / batch_size) / num_clients) / num_steps_until_rtc_check
        steps_in_one_epoch = ((n_train / batch_size) / num_clients)

    # 2. Federated Dataset creation
    federated_ds, test_ds = load_federated_data_fn(
        num_clients, batch_size, num_steps_until_rtc_check, bias=bias, seed=seed
    )

    # 3. Models creation
    server_cnn = server_compile_and_build_model_fn()
    client_cnns = [client_compile_and_build_model_fn() for _ in range(num_clients)]

    epoch_metrics_list = None
    sketch_width, sketch_depth = None, None

    # 4. Simulation

    # For `synchronous`, `naive`, `linear`, `sketch`
    # server_compile_and_build_model_fn , client_compile_and_build_model_fn are the same func.
    compile_and_build_model_fn = client_compile_and_build_model_fn

    if fda_name == "synchronous":
        epoch_metrics_list = synchronous_federated_simulation(
            test_ds, federated_ds, server_cnn, client_cnns, num_epochs,
            fda_steps_in_one_epoch, compile_and_build_model_fn
        )

    elif not per_layer:
        if fda_name == "naive":
            epoch_metrics_list = naive_federated_simulation(
                test_ds, federated_ds, server_cnn, client_cnns, num_epochs, theta,
                fda_steps_in_one_epoch, compile_and_build_model_fn, aggr_scheme
            )

        if fda_name == "linear":
            epoch_metrics_list = linear_federated_simulation(
                test_ds, federated_ds, server_cnn, client_cnns, num_epochs, theta,
                fda_steps_in_one_epoch, compile_and_build_model_fn, aggr_scheme
            )

        if fda_name == "sketch":
            sketch_width, sketch_depth = 250, 5

            if nn_name in ['EfficientNetV2L', 'ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtXLarge']:
                # ams_sketch = AmsSketch(save_on_cpu=True, use_other_gpu='/device:GPU:1', chunk_size=100_000_000)
                ams_sketch = AmsSketch(width=sketch_width, depth=sketch_depth, save_on_cpu=True, chunk_size=25_000_000)
                # ams_sketch = None
            else:
                ams_sketch = AmsSketch(width=sketch_width, depth=sketch_depth)

            epoch_metrics_list = sketch_federated_simulation(
                test_ds, federated_ds, server_cnn, client_cnns, num_epochs, theta, fda_steps_in_one_epoch,
                compile_and_build_model_fn, ams_sketch, 1. / sqrt(sketch_width),
                aggr_scheme
            )

        if fda_name == "gm":
            epoch_metrics_list = gm_federated_simulation(
                test_ds, federated_ds, server_cnn, client_cnns, num_epochs, theta,
                fda_steps_in_one_epoch, compile_and_build_model_fn, aggr_scheme
            )

        if fda_name in ["FedAdam", "FedAvgM"]:
            epoch_metrics_list = fed_opt_simulation(
                test_ds, federated_ds, server_cnn, client_cnns, num_epochs, steps_in_one_epoch
            )
    else:

        trainable_layers_indices = server_cnn.get_trainable_layers_indices()
        num_weights_per_layer = [len(v) for v in server_cnn.per_layer_trainable_vars_as_vector()]
        total_num_weights = sum(num_weights_per_layer)
        # percent-wise (on num. of weights) thetas per layer
        thetas = [theta * (num_weights / total_num_weights) for num_weights in num_weights_per_layer]

        if fda_name == "naive":
            epoch_metrics_list = naive_federated_simulation_per_layer(
                test_ds, federated_ds, server_cnn, client_cnns, num_epochs, thetas,
                fda_steps_in_one_epoch, compile_and_build_model_fn, aggr_scheme, trainable_layers_indices
            )

    # 5. Create Test ID
    test_id = TestId(
        ds_name, bias, aggr_scheme, fda_name, num_clients, batch_size, num_steps_until_rtc_check, theta, nn_name,
        count_weights(server_cnn), sketch_width, sketch_depth
    )

    # 6. Extend the metrics with Test ID
    epoch_metrics_with_test_id_list = process_metrics_with_test_id(
        epoch_metrics_list, test_id
    )
    
    return epoch_metrics_with_test_id_list
