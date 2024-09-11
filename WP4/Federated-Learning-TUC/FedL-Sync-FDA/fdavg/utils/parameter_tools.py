from fdavg.data.mnist import MNIST_N_TRAIN, MNIST_CNN_BATCH_INPUT, MNIST_CNN_INPUT_RESHAPE, mnist_load_federated_data
from fdavg.data.cifar10 import CIFAR10_N_TRAIN, CIFAR10_CNN_BATCH_INPUT, cifar10_load_federated_data
from fdavg.data.cifar100 import CIFAR100_N_TRAIN, CIFAR100_CNN_BATCH_INPUT, cifar100_load_federated_data
from fdavg.models.lenet5 import get_compiled_and_built_lenet
from fdavg.models.advanced_cnn import get_compiled_and_built_advanced_cnn
from fdavg.models.dense_net import get_compiled_and_built_densenet

from functools import partial

import tensorflow as tf


def derive_params(nn_name, ds_name, batch_size, num_clients, num_epochs, fda_name, **kwargs):
    """
    - MNIST dataset
        1. `gm`, `sketch`, `naive`, `linear` use Adam optimizer with default params on all clients, and the server has
            no optimizer at all. We put an optimizer though in order to be able to .compile the model. We do not use
            the server optimizer at all.
        2. `FedAdam` uses SGD with learning_rate 0.01 on clients, and the server USES an optimizer. The server uses
            Adam with the defaults, utilizing the pseudo-gradient average drifts.

    - CIFAR-10
        1. `gm`, `sketch`, `naive`, `linear` use SGD with Nesterov momentum b=0.9 with a learning rate of 0.1 on all
            clients. Server does NOT use an optimizer, we just put one there so that we can .compile.
        2. `FedAvgM` uses SGD with learning_rate=0.0316 on all clients and server USES an optimizer. The server uses
            SGD with momentum b=0.9 with learning_rate=0.316.

    In this function we aim to build generic partial functions to call from `simulation.py`. We do the dirty work here
    with all the if statements. What follows is ugly...
    """

    derived_params = {}

    # Only for MNIST dataset
    if ds_name == 'MNIST':

        derived_params['load_federated_data_fn'] = mnist_load_federated_data
        derived_params['n_train'] = MNIST_N_TRAIN

        get_compiled_and_build_model = None

        if nn_name == 'AdvancedCNN':
            get_compiled_and_build_model = get_compiled_and_built_advanced_cnn
        if nn_name == 'LeNet-5':
            get_compiled_and_build_model = get_compiled_and_built_lenet

        if fda_name in ['synchronous', 'gm', 'naive', 'linear', 'sketch']:

            compile_and_build_model_fn = partial(
                get_compiled_and_build_model,
                cnn_batch_input=MNIST_CNN_BATCH_INPUT,
                cnn_input_reshape=MNIST_CNN_INPUT_RESHAPE,
                num_classes=10,
                optimizer_fn=tf.keras.optimizers.Adam
            )
            derived_params['server_compile_and_build_model_fn'] = compile_and_build_model_fn  # For .compile ONLY
            derived_params['client_compile_and_build_model_fn'] = compile_and_build_model_fn

        if fda_name == 'FedAdam':

            derived_params['server_compile_and_build_model_fn'] = partial(
                get_compiled_and_build_model,
                cnn_batch_input=MNIST_CNN_BATCH_INPUT,
                cnn_input_reshape=MNIST_CNN_INPUT_RESHAPE,
                num_classes=10,
                optimizer_fn=tf.keras.optimizers.Adam
            )

            derived_params['client_compile_and_build_model_fn'] = partial(
                get_compiled_and_build_model,
                cnn_batch_input=MNIST_CNN_BATCH_INPUT,
                cnn_input_reshape=MNIST_CNN_INPUT_RESHAPE,
                num_classes=10,
                optimizer_fn=tf.keras.optimizers.SGD
            )

    if ds_name == 'CIFAR10':

        derived_params['load_federated_data_fn'] = cifar10_load_federated_data
        derived_params['n_train'] = CIFAR10_N_TRAIN

        if nn_name in ['DenseNet121', 'DenseNet169', 'DenseNet201']:

            if fda_name in ['synchronous', 'gm', 'naive', 'linear', 'sketch']:

                optimizer_fn = partial(
                    tf.keras.optimizers.SGD,
                    learning_rate=0.1,
                    momentum=0.9,
                    nesterov=True
                )

                compile_and_build_model_fn = partial(
                    get_compiled_and_built_densenet,
                    name=nn_name,
                    cnn_batch_input=CIFAR10_CNN_BATCH_INPUT,
                    optimizer_fn=optimizer_fn
                )

                derived_params['server_compile_and_build_model_fn'] = compile_and_build_model_fn  # Only for .compile
                derived_params['client_compile_and_build_model_fn'] = compile_and_build_model_fn

        if fda_name == 'FedAvgM':

            server_optimizer_fn = partial(
                tf.keras.optimizers.SGD,
                learning_rate=0.316,
                momentum=0.9
            )

            client_optimizer_fn = partial(
                tf.keras.optimizers.SGD,
                learning_rate=0.0316
            )

            derived_params['server_compile_and_build_model_fn'] = partial(
                get_compiled_and_built_densenet,
                name=nn_name,
                cnn_batch_input=CIFAR10_CNN_BATCH_INPUT,
                optimizer_fn=server_optimizer_fn
            )

            derived_params['client_compile_and_build_model_fn'] = partial(
                get_compiled_and_built_densenet,
                name=nn_name,
                cnn_batch_input=CIFAR10_CNN_BATCH_INPUT,
                optimizer_fn=client_optimizer_fn
            )

    if ds_name == 'CIFAR-100':

        # Here because ARIS has 2.7
        from fdavg.models.convnext import get_compiled_and_built_convnext

        derived_params['load_federated_data_fn'] = cifar100_load_federated_data
        derived_params['n_train'] = CIFAR100_N_TRAIN

        if nn_name in ['ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtXLarge']:

            if fda_name in ['synchronous', 'gm', 'naive', 'linear', 'sketch']:
                optimizer_fn = partial(
                    tf.keras.optimizers.AdamW,
                    learning_rate=5e-5,
                    weight_decay=1e-8
                )

                compile_and_build_model_fn = partial(
                    get_compiled_and_built_convnext,
                    name=nn_name,
                    cnn_batch_input=CIFAR100_CNN_BATCH_INPUT,
                    optimizer_fn=optimizer_fn
                )

                derived_params[
                    'server_compile_and_build_model_fn'] = compile_and_build_model_fn  # Only for .compile
                derived_params['client_compile_and_build_model_fn'] = compile_and_build_model_fn

        if nn_name in ['EfficientNetB7', 'EfficientNetV2L']:

            if fda_name in ['synchronous', 'gm', 'naive', 'linear', 'sketch']:
                optimizer_fn = partial(
                    tf.keras.optimizers.SGD,
                    learning_rate=0.01,
                    momentum=0.9,
                    global_clipnorm=1.
                )

                # maybe change? It follows paper tho
                # https://arxiv.org/pdf/2104.00298#page=10&zoom=100,0,0
                # https://arxiv.org/pdf/2010.11929#page=13&zoom=100,144,854

                compile_and_build_model_fn = partial(
                    get_compiled_and_built_convnext,
                    name=nn_name,
                    cnn_batch_input=CIFAR100_CNN_BATCH_INPUT,
                    optimizer_fn=optimizer_fn
                )

                derived_params[
                    'server_compile_and_build_model_fn'] = compile_and_build_model_fn  # Only for .compile
                derived_params['client_compile_and_build_model_fn'] = compile_and_build_model_fn

    return derived_params
