import json
import os
import time

import services.service as service
from configuration import get_model_fixed_params
from libraries.models.wrappers.wayeb_wraper import WayebWrapper, create_params_dict_from_list


class Factory(service.Service):
    type = "factory"
    commands = "commandstopic"
    reports = "reportstopic"
    saved_models = "models"
    dataset_versions = "datasets"
    assembly_reports = "assembleddatasets"

    assembled_datasets_path = None
    current_dataset_info = None  # {"version": version, "path": assembled_dataset_path, "lines": lines}
    dataset_in_use = None
    model_name = None
    model_init_params = None
    command_queue = []
    model = None
    last_model_id = 0
    deletion_queue = []
    production_start = 0
    production_end = 0
    opt_model_params = []
    train_percent = 0.7
    cache = None
    multiplier = 1

    def __init__(self, configuration_file):
        super().__init__(configuration_file)
        self.model_name = self.service_parameters["model"]
        self.assembled_datasets_path = self.service_parameters["assembleddatasetspath"]
        self.saved_models_path = self.service_parameters["savedmodelspath"]
        self.model_init_params = get_model_fixed_params(self.service_parameters)
        self.optimisation_model_start_id = None
        self.model = self.initialise_model()
        self.train_percent = float(self.service_parameters["trainpercent"])
        if "time" in self.service_parameters:
            if self.service_parameters["time"] == "millisec":
                self.multiplier = 1000


    def run(self):
        """
        Run process of factory service. Factory does the following
            - consumes dataset version messages from collector
                - if new version received, factory locks relative
                  buckets, copies bucket data into a single file
                  unlocks buckets.
            - receives train messages from optimiser service, in which case
                retrains the model with best_params
            - receives optimisation messages
                - optimisation initialisation (lock current dataset so that it does not get deleted)
                - optimisation step
                - optimisation finalise
        """
        # latest dataset updating
        version_rows = self.consume(self.dataset_versions)
        if version_rows:
            dataset_info = version_rows[0]

            if self.current_dataset_info is not None:
                self.deletion_queue.append(self.current_dataset_info)

            self.current_dataset_info = dataset_info
            self.current_dataset_info = self.assemble_dataset(dataset_info)
            self.produce(self.assembly_reports, [dataset_info], keys=["version"])
            self.clean_old_datasets()

        command = self.get_command()
        if command is not None:

            print(command, flush=True)
            print(self.current_dataset_info, flush=True)

            command_type = command["type"]
            # reading commands from optimiser service
            if command_type == "train":
                # train
                self.production_start = int(time.time())
                model_params = json.loads(command["params"])["params"]
                train_dataset_path = self.current_dataset_info["path"]
                model_save_path = f"{self.saved_models_path}/{self.model_name}_{self.last_model_id}.model"
                self.model.train(model_params, train_dataset_path, model_save_path)
                self.production_end = int(time.time())
                production_time = (self.production_end - self.production_start) * self.multiplier

            # send reply to scheduler
                reply = {"reply_id": command["id"],
                         "params_dict": command["params"],
                         "model_id": self.last_model_id,
                         "metrics": ""}
                self.produce(self.reports, [reply], keys=["reply_id"])


                # send new model info
                model_record = {
                    "id": self.last_model_id,
                    "path": model_save_path,
                    "creation_method": "retrain",
                    "model_params": json.dumps(model_params),
                    "pt": production_time
                }
                self.produce(self.saved_models, [model_record], keys=["id"])
                self.last_model_id = self.last_model_id + 1
                pass

            if command_type == "opt_initialise":
                print("Optimisation starting...", flush=True)
                print(self.current_dataset_info, flush=True)
                self.dataset_in_use = self.current_dataset_info
                self.optimisation_model_start_id = self.last_model_id
                self.production_start = int(time.time())
                self.opt_model_params = []
                pass
            if command_type == "opt_step":
                # train and save
                params = json.loads(command["params"])["params"]
                params_dict = create_params_dict_from_list(params)
                self.opt_model_params.append(params_dict)
                model_save_path = f"{self.saved_models_path}/{self.model_name}_{self.last_model_id}.model"
                obj_params = self.model.train(params_dict, self.dataset_in_use["path"], model_save_path)
                # load and test
                obj_params["confidence"] = params_dict["confidence"]
                metrics = self.model.test(params_dict, self.dataset_in_use["path"], model_save_path, obj_params)
                metrics_json_string = json.dumps(metrics)
                reply = {"reply_id": command["id"],
                         "params_dict": json.dumps(params_dict),
                         "model_id": self.last_model_id,
                         "metrics": metrics_json_string}
                self.produce(self.reports, [reply], keys=["reply_id"])
                self.last_model_id = self.last_model_id + 1
                # pass results
                pass
            if command_type == "opt_finalise":
                # save best trained model
                best_iter_id = command["best_i"]
                best_model_params = self.opt_model_params[best_iter_id]
                train_dataset_path = self.dataset_in_use["path"]
                model_save_path = f"{self.saved_models_path}/{self.model_name}_{self.last_model_id}.model"

                print(best_model_params, flush=True)
                self.model.train(best_model_params, train_dataset_path, model_save_path)

                self.production_end = int(time.time())
                production_time = (self.production_end - self.production_start)*self.multiplier

                # send reply to scheduler
                reply = {"reply_id": command["id"],
                         "params_dict": json.dumps(best_model_params),
                         "model_id": self.last_model_id,
                         "metrics": ""}

                self.produce(self.reports, [reply], keys=["reply_id"])

                # send new model version
                model_record = {"id": self.last_model_id,
                                "path": model_save_path,
                                "creation_method": "optimisation",
                                "model_params": json.dumps(best_model_params),
                                "pt": production_time}
                self.produce(self.saved_models, [model_record], keys=["id"])
                self.last_model_id = self.last_model_id + 1

                # release resources
                self.dataset_in_use = None
                self.optimisation_model_start_id = None
                print("Optimisation ended...", flush=True)

            pass

        pass

    def assemble_dataset(self, dataset_info):
        """
        Given a dataset info record of the form:
        data_version_record = {
                    "path_prefix": self.storage_path + self.naming,
                    "buckets_range": [max(0, info['bucket_id'] - self.k), info['bucket_id']],
                    "version": self.dataset_version
                }
        It assembles all buckets into a single file to be used by the factory.

        :param dataset_info: a dataset info record
        :return: a dict with three fields: version (version of the dataset), path (path to the dataset)
         and lines (number of lines in dataset)
        """
        buckets_range = dataset_info["buckets_range"]
        buckets_path_prefix = dataset_info["path_prefix"]
        version = dataset_info["version"]

        # lock buckets
        for i in buckets_range:
            lockfilepath = buckets_path_prefix + f"{i}" + ".lock"
            with open(lockfilepath, 'w') as fp:
                pass

        # assemble complete
        assembled_dataset_path = self.assembled_datasets_path + f"assembled_dataset_version_{version}"
        assembled_dataset_path_train = self.assembled_datasets_path + f"assembled_dataset_version_{version}_train"
        assembled_dataset_path_test = self.assembled_datasets_path + f"assembled_dataset_version_{version}_test"
        lines = 0
        first_t = -1
        last_t = -1
        with open(assembled_dataset_path, "w") as out, \
                open(assembled_dataset_path_train, "w") as out_train, \
                open(assembled_dataset_path_test, "w") as out_test:
            for i in buckets_range:
                bucket_path = buckets_path_prefix + f"{i}"
                with open(bucket_path, 'r') as inp:
                    for line in inp:
                        record_timestamp = json.loads(line)["timestamp"]
                        if first_t < 0:
                            first_t = record_timestamp
                        last_t = record_timestamp
                        lines = lines + 1
                        out.write(line)

            # lines_train = lines * self.train_percent
            # with open(assembled_dataset_path, 'r') as inp:
            #     i = 0
            #     for line in inp:
            #         if i < lines_train:
            #             out_train.write(line)
            #         else:
            #             out_test.write(line)
            #         i = i + 1


        # assemble
        assembled_dataset_path = self.assembled_datasets_path + f"assembled_dataset_version_{version}"
        lines = 0
        with open(assembled_dataset_path, "w") as out:
            for i in buckets_range:
                bucket_path = buckets_path_prefix + f"{i}"
                with open(bucket_path, 'r') as inp:
                    for line in inp:
                        lines = lines + 1
                        out.write(line)

        # unlock buckets
        for i in buckets_range:
            lockfilepath = buckets_path_prefix + f"{i}" + ".lock"
            os.remove(lockfilepath)

        assembled_dataset_info = {"version": version,
                        "path": assembled_dataset_path,
                        "path_train": assembled_dataset_path_train,
                        "path_test": assembled_dataset_path_test,
                        "lines": lines,
                        "first_t": first_t,
                        "last_t": last_t}

        print(assembled_dataset_info, flush=True)
        return assembled_dataset_info

    def initialise_model(self):
        """
        Initialisation of model selected under self.model_name.
        Models must be initialised before trained/ or tested.

        :return: a model wrapper
        """
        model = None
        if self.model_name == "wayeb":
            model = WayebWrapper(self.model_init_params)
        return model

    def get_command(self):
        """
        If a dataset is not available commands are added to a queue.
        If a dataset exists and the queue is not empty provide commands from the queue.
        if a dataset exists and the queue is empty return consumed command.
        :return:
        """
        command_rows = self.consume(self.commands)
        if command_rows:
            command = command_rows[0]
            if self.correct_dataset_available(command["timestamp"]) and len(self.command_queue) == 0:
                return command
            else:
                self.command_queue.append(command)

        if  len(self.command_queue) > 0 and self.correct_dataset_available(self.command_queue[0]["timestamp"]):
            return self.command_queue.pop(0)
        else:
            return None

    def correct_dataset_available(self, timestamp):
        if self.current_dataset_info is None:
            return False

        # dirty fix
        # TODO fix by transmitting of timestamp of first record
        #  of next bucket when sending dataset version
        if (timestamp-600) > self.current_dataset_info["last_t"]:
            return False

        return True

    def clean_old_datasets(self):
        """
        Deletes datasets in deletion queue if they are not in use.
        """
        new_queue = []
        for dataset in self.deletion_queue:
            if dataset != self.dataset_in_use:
                os.remove(dataset["path"])
                os.remove(dataset["path_train"])
                os.remove(dataset["path_test"])
            else:
                new_queue.append(dataset)
        self.deletion_queue = new_queue

    def cleanup(self):
        """
        Factory process cleanup.
        """
        if self.current_dataset_info is not None:
            os.remove(self.current_dataset_info["path"])
            os.remove(self.current_dataset_info["path_train"])
            os.remove(self.current_dataset_info["path_test"])

        for dataset in self.deletion_queue:
            os.remove(dataset["path"])
            os.remove(dataset["path_train"])
            os.remove(dataset["path_test"])
