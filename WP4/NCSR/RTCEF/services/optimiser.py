import json

import services.service as service
from configuration import get_parameters_domain, get_model_fixed_params
from libraries.optimisers.skopt_wrapper import SkoptWrapper


class Optimiser(service.Service):
    type = "optimiser"
    parameters_domain = {}
    current_best_params = {}
    instructions = "instructions"
    commands = "commands"
    reports = "reports"
    syncing = "enginesync"
    train_counter_id = 0
    optimise_counter_id = 0
    command_id = 0
    samples_folder = None
    logs_folder = None
    pkl_folder = None
    sync_engine = False
    opt_wrapper = None
    reuse = False
    initial_opt_model_path = None

    def __init__(self, configuration_file):
        super().__init__(configuration_file)

        self.library = self.service_parameters["library"]
        self.parameters_domain = get_parameters_domain(self.service_parameters)
        self.n_init_bench = int(self.service_parameters["ninitbench"])
        self.n_total_evals = int(self.service_parameters["ntotalevals"])
        self.acquisition_function = self.service_parameters["acquisitionfunction"]
        self.samples_folder = self.service_parameters["optimisationsamples"]
        self.logs_folder = self.service_parameters["optimisationlogs"]
        self.pkl_folder = self.service_parameters["optimisationmodels"]
        self.seed = int(self.service_parameters["seed"])
        self.sync_engine = self.service_parameters["syncengine"] == "true"
        self.current_best_params = get_model_fixed_params(self.service_parameters)
        if self.service_parameters["reuse"] == "true":
            self.reuse = True
        self.keep_n_evals = int(self.service_parameters["keepnevals"])
        if "initialmodel" in self.service_parameters:
            self.initial_opt_model_path = self.service_parameters["initialmodel"]

    def run(self):
        """
        Main service loop:
            The optimiser consumes instructions from observer,
                and either sends a training message
                or, begins an optimisation procedure.
        """
        rows = self.consume(self.instructions)
        for row in rows:
            instruction = row["instruction"]
            timestamp = row["timestamp"]
            if instruction == "train":
                if self.sync_engine:
                    sync_record = {
                        "timestamp": timestamp,
                        "type": "pause",
                        "model_id": -1
                    }
                    self.produce(self.syncing, [sync_record], ["timestamp", "type"])

                model_id = self.train(self.current_best_params, timestamp)

                if self.sync_engine:
                    sync_record = {
                        "timestamp": timestamp,
                        "type": "play",
                        "model_id": model_id
                    }
                    self.produce(self.syncing, [sync_record], ["timestamp", "type"])
            else:
                if self.sync_engine:
                    sync_record = {
                        "timestamp": timestamp,
                        "type": "pause",
                        "model_id": -1
                    }
                    self.produce(self.syncing, [sync_record], ["timestamp", "type"])

                model_id = self.optimise(timestamp)

                if self.sync_engine:
                    sync_record = {
                        "timestamp": timestamp,
                        "type": "play",
                        "model_id": model_id
                    }
                    self.produce(self.syncing, [sync_record], ["timestamp", "type"])

    def train(self, parameters, timestamp):
        """
        Send a train command to the factory, with the given model parameters.
        :param parameters: a dict with the parameters to send to the factory for model training.
        """
        train_command = self.get_command("train", timestamp, parameters)
        self.produce(self.commands, [train_command], keys=["id"])
        #  Consume training completion
        while True:
            rows = self.consume(self.reports)
            for row in rows:
                # reply_id = row["reply_id"]
                model_id = row["model_id"]
                return model_id
                pass

    def optimise(self, timestamp):
        """
        Start an optimisation procedure. During this process the optimiser
        initialises an optimisation procedure with the factory, then sends
        training commands (optimisation steps), receives results, updates
        optimisation model and finalises the optimisation procedure.

        """
        name = f"optimisation_{self.optimise_counter_id}"

        logpath = self.logs_folder + name + ".log"
        pklpath = self.pkl_folder + name + ".log"
        samplespath = self.samples_folder + name + ".log"

        if self.opt_wrapper is None or self.reuse == False:
            self.opt_wrapper = SkoptWrapper(self.seed + self.optimise_counter_id,
                                            logpath,
                                            pklpath,
                                            samplespath,
                                            self.parameters_domain,
                                            self.keep_n_evals,
                                            self.initial_opt_model_path)

        self.opt_wrapper.initialise_optimisation(self.n_init_bench, self.n_total_evals, self.acquisition_function,
                                                 reuse=True)

        self.opt_wrapper.step_by_step_optimisation()

        command = self.get_command("opt_initialise", timestamp)
        self.produce(self.commands, [command], keys=["id"])

        while True:
            # get parameters to send
            params = self.opt_wrapper.get_current_step_params()
            params2send = params
            # send parameters to factory
            command = self.get_command("opt_step", timestamp, params2send)
            self.produce(self.commands, [command], keys=["id"])

            metrics = None

            # consume messages for results
            while True:
                rows = self.consume(self.reports)
                if rows:
                    assert len(rows) == 1
                    for row in rows:
                        result = row
                        metrics = json.loads(result["metrics"])
                    break

            assert metrics is not None
            more_points_left = self.opt_wrapper.optimisation_step(params, metrics)
            if not more_points_left:
                break

        optimisation_results = self.opt_wrapper.finalise()
        best_i = optimisation_results["i"]

        # send finalise command to factory
        command = self.get_command("opt_finalise", timestamp, params=None, best_i=best_i)
        self.produce(self.commands, [command], keys=["id"])
        while True:
            rows = self.consume(self.reports)
            for row in rows:
                # reply_id = row["reply_id"]
                model_id = row["model_id"]
                self.current_best_params = json.loads(row["params_dict"])
                return model_id
                pass

    def get_command(self, instruction, timestamp, params=None, best_i=-1):
        """
        Creates a command in the appropriate format, and updates the ids (optimisation_id, train_id, and cid)

        :param timestamp: timestamp corresponding to observer instruction
        :param best_i: iteration number of the iteration with the best
            objective
        :param instruction: type of command to create
        :param params: params should be provided in case they are need in command
        :return: a dict of the form:
            {"train_id": train_id,  the id of the train command
             "optimisation_id": optimisation_id, the id of the optimisation command
             "id": cid, the command id
             "type": instruction, type of the instruction = {train, opt_initialise, opt_step, opt_finalise}
             "params": params parameters to use for model training
             }
        """
        train_id = -1
        optimisation_id = -1

        cid = self.command_id
        params_json_string = json.dumps({"params": params})
        if instruction == "train":
            train_id = self.train_counter_id
            self.train_counter_id = self.train_counter_id + 1
        if "opt" in instruction:
            optimisation_id = self.optimise_counter_id
            if instruction == "opt_finalise":
                self.optimise_counter_id = self.optimise_counter_id + 1
        msg = {
            "train_id": train_id,
            "optimisation_id": optimisation_id,
            "id": cid,
            "timestamp": timestamp,
            "type": instruction,
            "params": params_json_string,
            "best_i": best_i
        }
        self.command_id = cid + 1
        return msg
