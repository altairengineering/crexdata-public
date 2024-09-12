import os
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# Get the directory of the current script (main.py)
script_dir = os.path.dirname(os.path.realpath(__file__))
tmp_dir = '../metrics/tmp'
epoch_metrics_path = os.path.normpath(os.path.join(script_dir, f'{tmp_dir}/epoch_metrics'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--comb_file_id', type=int, default=-1, help="The combinations prefix, i.e., <PREFIX>.json")
    parser.add_argument('--sim_id', type=int, help="The Sim ID (0, 1, 2, ...)")
    parser.add_argument('--gpu_id', type=str, help="The GPU ID that the simulation will use.")
    parser.add_argument('--gpu_mem', type=int, default=-1,
                        help="The maximum GPU memory. If not given we dynamically allocate.")
    parser.add_argument('--slurm', action='store_true', help="Use if we are in SLURM HPC env.")
    parser.add_argument('--kafka', action='store_true',
                        help="Use if we are reading hyper-parameters from Kafka.")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        if args.gpu_mem == -1:
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_mem)]
                )
    else:
        print("No GPUs Available!")

    import time
    import pandas as pd

    from fdavg.strategies.simulation import single_simulation
    from fdavg.utils.read_combinations import get_test_hyper_parameters
    from fdavg.utils.pretty_printers import print_finish_testing_info, print_current_test_info
    from fdavg.utils.parameter_tools import derive_params

    if not args.kafka:
        hyperparameters = get_test_hyper_parameters(f'{args.comb_file_id}', args.sim_id)
    else:
        from fdavg.utils.read_combinations import kafka_get_test_hyper_parameters
        hyperparameters = kafka_get_test_hyper_parameters()

    # 1. Derive parameters
    derived_params = derive_params(**hyperparameters)

    # 2. Metrics
    all_epoch_metrics = []

    # Start timer
    start_time = time.time()

    # Print current test info
    print_current_test_info(**hyperparameters)

    # 3. Run simulation
    epoch_metrics_with_test_id_list = single_simulation(**derived_params, **hyperparameters)

    # 4. Append metrics
    all_epoch_metrics.extend(epoch_metrics_with_test_id_list)

    # Simulation ended
    print_finish_testing_info(start_time=start_time, end_time=time.time(), **hyperparameters)

    # 5. Save metrics
    epoch_metrics_df = pd.DataFrame(all_epoch_metrics)
    
    if args.slurm:
        epoch_metrics_df.to_csv(f"{epoch_metrics_path}/{hyperparameters['test_id']}.csv", index=False)
    else:
        epoch_metrics_df.to_parquet(f"{epoch_metrics_path}/{hyperparameters['test_id']}.parquet")
