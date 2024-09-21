import json

import numpy as np

import services.service as service
from configuration import get_service_parameters


class Observer(service.Service):
    type = "observer"
    consuming_scores_name = "modelscores"
    producing_instr_name = "instructions"

    method = None
    k = 2
    train_difference = None
    opt_difference = None
    score_field = None
    metrics_type = "batch_metrics"
    scores = []
    id = 0
    last_model_version = -1  # initial model id
    recent_train_threshold = 3
    grace_period = 0
    low_score_threshold = None
    recent_train = False
    start = 0
    low_score_threshold = 0.1

    def __init__(self, configuration_file):
        """
        Initialisation of the observer service. The observer service,
        monitors performance reports and according to the selected
        method produces a "train" or "optimise" instruction for
        the optimiser service.

        :param configuration_file: path to the configuration file.
         Configuration file variables:
            * "method" should be one of the following
                - difference
                    if method is difference additionally the following must be provided
                        traind: train difference threshold
                        hoptd: hyperparameter optimisation difference threshold
                - ...
            * "scorefield" name of the scorefield to use from messages read from
               reports topic
            * "timefield" name of the field to use for timestamps
        """
        super().__init__(configuration_file)
        service_parameters = get_service_parameters(self.config, self.type)

        self.method = service_parameters["method"]
        self.k = int(service_parameters["k"])
        if self.method == "difference":
            assert self.k == 2
        self.train_difference = float(service_parameters["traind"])
        self.opt_difference = float(service_parameters["hoptd"])
        self.score_field = service_parameters["scorefield"]
        self.time_field = service_parameters["timefield"]
        self.metrics_type = service_parameters["metricstype"]
        self.slope_threshold = float(service_parameters["slopethreshold"])
        self.recent_train_threshold = int(service_parameters["recenttrainthreshold"])

        if "lowscorethreshold" in service_parameters:
            self.low_score_threshold = float(service_parameters["lowscorethreshold"])

    def run(self):
        """
        Run the service. While running, the observer service reads
        messages from the reports topic and using the selected method
        (e.g., difference) produces instructions messages to be read
        by the optimiser service
        """
        rows = self.consume(self.consuming_scores_name)
        for row in rows:
            if self.recent_train:
                self.grace_period = self.grace_period - 1
                if self.grace_period == 0:
                    self.recent_train = False
            # model_id = int(row["model_version"])
            # if model_id != self.last_model_version:
            #     self.last_model_version = model_id
            #     self.scores = []
            self.scores.append(json.loads(row[self.metrics_type]))
            if len(self.scores) > self.k:
                self.scores.pop(0)
            print(row)
            assessment_result = self.assess(self.scores)
            if assessment_result != "noop":
                train_instruction = {
                    "id": self.id,
                    "timestamp": row[self.time_field],
                    "instruction": assessment_result
                }
                self.id = self.id + 1
                print(train_instruction)
                self.produce(self.producing_instr_name, [train_instruction])
            pass

    def assess(self, scores):
        """
        based on the selected assessment methodology the method returns
        the appropriate instruction type i.e., train/optimise/noop.

        :param scores: a list of scores
        :return: a string that has three possible values:
            "train": assessment says to retrain
            "optimise": assessment says to optimise
            "noop": no operation is required
        """
        cur = scores[-1][self.score_field]
        extra = False
        print(extra)
        if self.low_score_threshold is not None:
            if cur < self.low_score_threshold:
                extra = True
        #return "noop"
        if extra:
            return "optimise"

        if len(self.scores) > 1:
            if self.method == "difference":
                if self.len(self.scores) >= 2:
                    prev = scores[0][self.score_field]
                    cur = scores[1][self.score_field]
                    diff = prev - cur
                    if self.train_difference < diff < self.opt_difference:
                        return "train"
                    elif diff > self.opt_difference:
                        return "optimise"

            if self.method == "regression":
                rr = self.k
                if len(self.scores) < self.k:
                    rr = len(self.scores)
                x = [i for i in range(len(self.scores) - rr, len(self.scores))]
                y = [self.scores[i][self.score_field] for i in range(len(self.scores) - rr, len(self.scores))]
                coef = np.polyfit(x, y, 1)

                print(self.scores)
                print(extra)
                print([x, y], flush=True)
                print(coef, flush=True)

                if coef[0] < self.slope_threshold and self.recent_train:
                    self.recent_train = True
                    self.grace_period = self.recent_train_threshold
                    return "optimise"
                elif (coef[0] < self.slope_threshold) and not self.recent_train:
                    self.recent_train = True
                    self.grace_period = self.recent_train_threshold
                    return "train"

            if self.method == "hybrid":
                prev = scores[0][self.score_field]
                cur = scores[1][self.score_field]
                diff = prev - cur

                rr = self.k
                if len(self.scores) < self.k:
                    rr = len(self.scores)
                x = [i for i in range(len(self.scores) - rr, len(self.scores))]
                y = [self.scores[i][self.score_field] for i in range(len(self.scores) - rr, len(self.scores))]
                coef = np.polyfit(x, y, 1)

                if ((coef[0] < self.slope_threshold) and self.recent_train == True) or extra:
                    self.recent_train = False
                    return "optimise"
                elif (coef[0] < self.slope_threshold) and self.recent_train == False:
                    self.recent_train = True
                    self.grace_period = self.recent_train_threshold
                    return "train"

        return "noop"
