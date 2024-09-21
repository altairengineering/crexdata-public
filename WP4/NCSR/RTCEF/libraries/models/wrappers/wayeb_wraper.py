from math import tanh

import libraries.models.utils as utils


def create_params_dict_from_list(params_list):
    """
    Creates a dict with the parameters of wayeb using params_list.

    :param params_list: a list containing, order, confidence threshold, p min, and gamma
    :return: a dict of the form {
        'order': params_list[0],
        'runConfidenceThreshold': params_list[1],
        'pMin': params_list[2],
        'gamma': params_list[3],
    }
    """
    return {
        'order': int(params_list[1]),
        'confidence': float(params_list[0]),
        'pmin': float(params_list[2]),
        'gamma': float(params_list[3]),
    }


class WayebWrapper:
    weights = [0.5, 0.5]
    threshold_time = 6000
    domain = None
    gateways = {}
    runs = {}
    pattern_path = None
    declaration_path = None
    multicall = 0
    objective_func = None
    max_order = 5

    # def __init__(self, gate_runs, k_val, weights=None, threshold_time=None):
    def __init__(self, init_params_dict):
        self.python_port = int(init_params_dict["python_port"])
        self.java_port = int(init_params_dict["java_port"])
        self.k_val = int(init_params_dict["k_val"])

        if "weight_0" in init_params_dict and "weight_1" in init_params_dict:
            weight0 = float(init_params_dict["weight_0"])
            weight1 = float(init_params_dict["weight_1"])
            self.weights = [weight0, weight1]
        if "threshold_time" in init_params_dict:
            self.threshold_time = int(init_params_dict["threshold_time"])
        if "max_order" in init_params_dict:
            self.max_order = int(init_params_dict["max_order"])

        self.pattern_path = init_params_dict["pattern_path"]
        self.declaration_path = init_params_dict["declaration_path"]
        self.domain = init_params_dict["domain"]
        self.objective_func = init_params_dict["objective_func"]
        self.distance_min = init_params_dict["distance_min"]
        self.distance_max = init_params_dict["distance_max"]

        self.gateway = utils.build_gateway(self.python_port, self.java_port)
        self.run = self.gateway.entry_point

    def train(self, params, train_set_path, model_save_path):
        """

        :param params: params dict to use for training
        :param train_set_path: path to the training set
        :param model_save_path: path to save the spst model
        :return: a dict with arguments that are needed for
        evaluating the objective function (method is either
         "nt"/"comb" therefore it should contain "tt":training time
         or "order": order used for model)
        """
        run = self.run

        extra_args = run.trainAndSave(train_set_path, self.pattern_path,
                                      self.declaration_path, model_save_path,
                                      self.domain, self.distance_min, self.distance_max,
                                      params['order'], params['pmin'],
                                      params['gamma'], self.objective_func)
        if self.objective_func == "comb":
            return {"tt": extra_args[0]}
        elif self.objective_func == "nt":
            return {"order": extra_args[0]}
        pass

    def test(self, params, test_set_path, model_save_path, objective_args):
        """
        performs testing of model under "model_save_path"

        :param params: parameter dict
        :param test_set_path: dataset path to use for testing
        :param model_save_path: saved model path
        :param objective_args: extra arguments to use in the objective function
        :return: metrics dict of the form:
            {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "mcc": mcc, "f_val": f_val}
        """
        run = self.run

        metrics = run.loadAndTest(test_set_path, model_save_path,
                                  self.domain, self.distance_min,
                                  self.distance_max, params['confidence'])

        tp = metrics[0]
        tn = metrics[1]
        fp = metrics[2]
        fn = metrics[3]

        mcc = utils.calculate_mcc(tp, tn, fp, fn)

        metrics_dict = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "mcc": mcc}

        objective_args["threshold_time"] = self.threshold_time
        objective_args["max_order"] = self.max_order

        f_val = evaluate_score(self.weights, metrics_dict, self.objective_func, objective_args)

        metrics_dict["f_val"] = f_val

        return metrics_dict

    def close_gateway(self):
        self.gateway.shutdown()


def evaluate_score(weights, metrics, objective_func, objective_args, negative=-1):
    """
    Calculates the objective function value.

    :param weights: list with two weights.
    :param metrics: metrics to be used in the objective (in this case it must have mcc)
    :param objective_func: either "comb" or "nt"
    :param objective_args: extra arguments i.e., either tt or order
    :param negative: -1 (default) or 1, if -1 objective is multiplied by -1
    :return:
    """
    if objective_func == "comb":
        return negative * (weights[0] * metrics["mcc"] - weights[1] * tanh((objective_args["tt"] /
                                                                            objective_args["threshold_time"]) - 1))
    elif objective_func == "nt":
         return negative * (weights[0] * metrics["mcc"] - weights[1] * tanh((objective_args["order"] /
                                                                            objective_args["max_order"]) - 1))
        #return negative * (weights[0] * metrics["mcc"] + weights[1] * objective_args["confidence"] )
