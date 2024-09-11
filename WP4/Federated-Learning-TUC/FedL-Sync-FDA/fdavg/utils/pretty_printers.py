
def print_current_test_info(ds_name, bias, fda_name, num_clients, batch_size, num_epochs,
                            num_steps_until_rtc_check, nn_name, bench_test, theta, aggr_scheme, per_layer, **kwargs):
    nn_name = nn_name if nn_name != 'AdvancedCNN' else 'VGG16*'
    print()
    print(f"------------ Current Test : ------------")
    print(f"Dataset name : {ds_name}")
    print(f"Bias : {bias}")
    print(f"Bench test : {bench_test}")
    print(f"NN name : {nn_name}")
    print(f"FDA name : {fda_name}")
    print(f"Num Clients : {num_clients}")
    print(f"Batch size : {batch_size}")
    print(f"Num Epochs : {num_epochs}")
    print(f"Number of steps until we check RTC : {num_steps_until_rtc_check}")
    print(f"Theta : {theta}")
    print(f"Aggr. Scheme : {aggr_scheme}")
    print(f"Per-Layer training : {per_layer}")
    print("-----------------------------------------")
    print()
    
    
def print_finish_testing_info(start_time, end_time, ds_name, bias, fda_name, num_clients, batch_size, num_epochs,
                              num_steps_until_rtc_check, nn_name, bench_test, theta, aggr_scheme, per_layer, **kwargs):
    nn_name = nn_name if nn_name != 'AdvancedCNN' else 'VGG16*'
    print()
    print(f"------------ Finished Testing : ------------")
    print(f"Dataset name : {ds_name}")
    print(f"Bias : {bias}")
    print(f"Bench test : {bench_test}")
    print(f"NN name : {nn_name}")
    print(f"FDA name : {fda_name}")
    print(f"Num Clients : {num_clients}")
    print(f"Batch size : {batch_size}")
    print(f"Num Epochs : {num_epochs}")
    print(f"Number of steps until we check RTC : {num_steps_until_rtc_check}")
    print(f"Theta : {theta}")
    print(f"Aggr. Scheme : {aggr_scheme}")
    print(f"Per-Layer training : {per_layer}")
    print(f"Total simulation time: {end_time-start_time} sec")
    print("-----------------------------------------")
    print()


def print_epoch_metrics(epoch_metrics):
    print()
    print(f"---------- Epoch {epoch_metrics.epoch} Finished : -----------")
    print(f"Total Synchronizations : {epoch_metrics.total_rounds}")
    print(f"Test Accuracy : {epoch_metrics.accuracy:.4f}")
    print(f"Train Accuracy : {epoch_metrics.train_accuracy:.4f}")
    print("------------------------------------------")
    print()
