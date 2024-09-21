import numpy as np
from skopt.learning import GaussianProcessRegressor
from skopt import Optimizer, expected_minimum
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real, Integer

import pickle
from math import inf
import logging


class SkoptWrapper:
    gpr = None
    opt = None
    logger = None
    n_initial_points = None
    n_total_points = None
    opt_res = None
    ssbv = {}

    def __init__(self, seed,
                 logfile,
                 optimiser_file,
                 samples_file,
                 dimensions,
                 keep_n_evals=-1,
                 model_path=None,
                 logprinter=None,
                 csvprinter=None):
        """

        :param seed: integer to be used for seed
        :param logfile: logfile path
        :param optimiser_file: file to print optimisation results
        :param samples_file: file to print samples
        :param dimensions: dimensions dict (check skopt docs)
        :param keep_n_evals: default n=-1, if n > 0, and reuse=true in initialise_optimisation,
               n samples from previous run are kept.
        :param model_path: path to an optimiser pkl to load (optional)
        :param logprinter: logprinter function (optional)
        :param csvprinter: csvprinter function (optional)
        """
        self.logfile = logfile
        self.optimiser_file = optimiser_file
        self.samples_file = samples_file
        self.dimensions = get_dimensions_from_dict(dimensions)
        self.params_metrics_printer = logprinter
        self.csv_printer = csvprinter
        self.keep_n_evals = keep_n_evals

        FORMAT = '%(asctime)s - %(message)s'
        logging.basicConfig(format=FORMAT, filename=logfile, level=logging.INFO)
        self.logger = logging.getLogger("skopt")
        np.random.seed(seed)
        self.seed = seed
        self.model_path = model_path
        self.start_loading=True

    def initialise_optimisation(self, n_initial_points, n_total_points, acq_function, reuse=False):
        """
        Initialisation of an optimisation procedure. This function should be called before each optimisation
        procedure. If reuse is true, and initialise_optimisation has been called before the previous optimiser
        will be used again. If a model path has been provided during class object construction, then an optimiser
        is loaded from file.

        :param n_initial_points: sampling points
        :param n_total_points: number of all points
        :param acq_function: acquisition function to use (see skopt docs)
        :param reuse (boolean) old model from file or from previous call.
        """
        # print(f"Reuse: {reuse}")
        if self.opt is None or not reuse:
            self.gpr = GaussianProcessRegressor(kernel=1.0 * Matern(length_scale=[0.2, 1.0, 0.0001, 0.0001],
                                                                    nu=1.5,
                                                                    length_scale_bounds=(1e-5, 1.0)),
                                                    n_restarts_optimizer=2,
                                                    normalize_y=True,
                                                    # alpha=(1e-10,
                                                    noise=0.0)
            initial = n_initial_points
            if self.keep_n_evals > 0:
                initial = self.keep_n_evals
            self.opt = Optimizer(
                dimensions=self.dimensions,
                acq_func=acq_function,
                base_estimator=self.gpr,
                n_initial_points=initial,
                random_state=self.seed,
                acq_optimizer='auto',
                initial_point_generator="lhs"
            )

        self.n_initial_points = n_initial_points
        self.n_total_points = n_total_points
        if reuse:
            if self.model_path is not None and self.start_loading:
                print("Reusing optimiser from file....")
                with open(self.model_path, 'rb') as f:
                    old_opt = pickle.load(f)
                    x_iters = [[x[1], x[0], x[2], x[3]] for x in old_opt.x_iters]
                    y_vals = old_opt.func_vals

                    for iter in range(len(x_iters)):
                        self.opt.tell(x_iters[iter], y_vals[iter])

                    if self.keep_n_evals > 0:
                        to_remove = len(self.opt.Xi) - self.keep_n_evals
                        self.remove_samples(to_remove, "oldest")
                        self.n_total_points = self.n_total_points - self.keep_n_evals
                        self.n_initial_points = 0

            elif not self.start_loading:
                print("Reusing optimiser from previous run....")
                if self.keep_n_evals > 0:
                    to_remove = len(self.opt.Xi) - self.keep_n_evals
                    self.remove_samples(to_remove, "random")

                    self.n_total_points = self.n_total_points - self.keep_n_evals
                    self.n_initial_points = 0

        if self.start_loading:
            self.start_loading = False

    def remove_samples(self, to_remove, method="random"):
        """
        Remove samples from optimiser.

        :param to_remove: number of points to remove
        :param method: random or oldest
        """
        N = len(self.opt.Xi)
        indexes_to_remove = []
        if to_remove > 0:
            if method == "oldest":
                indexes_to_remove = [i for i in range(to_remove)]
            if method == "random":
                indexes_to_remove = np.random.choice([i for i in range(N)], to_remove, replace=False)

            Xi=[]
            yi=[]
            for i in range(N):
                if i not in indexes_to_remove:
                    Xi.append(self.opt.Xi[i])
                    yi.append(self.opt.yi[i])
            self.opt.Xi = Xi
            self.opt.yi = yi

    def step_by_step_optimisation(self):
        """
        Initalisation of the variables used in step by step optimisation.
        """
        self.ssbv = {}
        self.ssbv["initial"] = self.n_initial_points
        self.ssbv["total"] = self.n_total_points
        self.ssbv["phase"] = "initial"
        self.ssbv["res"] = None
        self.ssbv["best_params"] = {"initial": None, "optimisation": None}
        self.ssbv["best_objective"] = {"initial": inf, "optimisation": inf}
        self.logger.info("Initialised step by step optimisation...")
        extra_points = self.ssbv['total'] - self.ssbv['initial']
        self.ssbv["i"] = 0
        self.ssbv["res"] = None
        self.ssbv["best_i"] = 0
        self.ssbv["best_obj"] = inf
        self.logger.info(
            f"Total points = {self.ssbv['total']}, initial points = {self.ssbv['initial']}, extra points = {extra_points}")
        pass

    def finalise(self):
        """
        Finalise optimisation.

        :return: A dict with the best parameters,the best objective and best iteration number e.g.
                {"params": self.ssbv["best_params"]["optimisation"],
                 "objective": self.ssbv['best_objective']["optimisation"],
                 "i": self.ssbv["best_i"]}
        """
        with open(self.samples_file, "a") as out:
            self.opt_res = self.ssbv["res"]
            self.logger.info("Finished optimisation loop.")
            self.logger.info(f"Best objective value from optimisation is:"
                             f" {self.ssbv['best_objective']['optimisation']}.")
            self.logger.info(f"Best parameters: {self.ssbv['best_params']['optimisation']}.")
            out.write(f"\n\n{self.ssbv['best_objective']} -- {self.ssbv['best_params']}\n")
        if self.ssbv['best_objective']["initial"] < self.ssbv['best_objective']["optimisation"]:
            return {"params": self.ssbv["best_params"]["initial"],
                    "objective": self.ssbv['best_objective']["initial"],
                    "i": self.ssbv["best_i"]}
        else:
            return {"params": self.ssbv["best_params"]["optimisation"],
                    "objective": self.ssbv['best_objective']["optimisation"],
                    "i": self.ssbv["best_i"]}

    def get_current_step_params(self):
        """

        :return: The parameters to be used in the next step
        """
        next_x = self.opt.ask()
        print(f"Iteration {self.ssbv['i']}, params: {next_x}")
        return next_x

    def optimisation_step(self, next_x, results):
        """
        Performs an optimisation step using the given params, and results produced
        with given params. Results should contain a field that has the name "f_val"
        :param next_x: params
        :param results: dict with results, it must contain the value of the
         objective under key "f_val".
        :return:
        """
        i = self.ssbv["i"]
    #    objective_value = -results["mcc"]# results["f_val"]
        objective_value = results["f_val"]
        print(f"Iteration {i}, f_val = {objective_value}")

        phase = self.ssbv["phase"]

        with open(self.samples_file, "a") as out:
            if self.params_metrics_printer is None:
                self.logger.info(f"Point {i}: parameters={next_x} objective value={objective_value}")
            else:
                self.logger.info(f"Point {i}: {self.params_metrics_printer(next_x, results)}")

            if self.csv_printer is None:
                pars = ",".join([str(par) for par in next_x])
                out.write(f"{phase},{i},{pars},{objective_value}\n")
            else:
                print(f"{phase},{i},{self.csv_printer(next_x, results)}\n")
                out.write(f"{phase},{i},{self.csv_printer(next_x, results)}\n")

            if objective_value < self.ssbv["best_obj"]:
                self.ssbv["best_i"] = i
                self.ssbv["best_obj"] = objective_value

            if objective_value < self.ssbv["best_objective"][phase]:
                self.ssbv["best_params"][phase] = next_x
                self.ssbv["best_objective"][phase] = objective_value

            self.ssbv["res"] = self.opt.tell(next_x, objective_value)
            self.ssbv["i"] = i + 1
            if self.ssbv["i"] == self.ssbv["initial"]:
                self.logger.info("Finished initial sampling.")
                self.logger.info(f"Best objective value from initial sampling is: "
                                 f"{self.ssbv['best_objective']['initial']}.")
                self.logger.info(f"Starting optimisation phase...")
                self.ssbv["phase"] = "optimisation"
            if self.ssbv["i"] == self.ssbv["total"]:
                return False
        return True

    def save(self):
        with open(self.optimiser_file, 'wb') as f:
            pickle.dump(self.opt_res, f)


def get_dimensions_from_dict(dimensions_dict):
    """
    Given a dimensions dict of the form:
    {
        "paramA": [minA, maxA],
        "paramB": [minB, maxB],
        ...
    }
    the function returns a dimension list in the form of skopt
    :param dimensions_dict: dict with param names as keys and values list with min and max param values.
    :return: a dimension list in the form of skopt
    """
    dimensions = []
    for dim in dimensions_dict:
        bound_min = dimensions_dict[dim][0]
        bound_max = dimensions_dict[dim][1]
        if isinstance(dimensions_dict[dim][0], int):
            dimensions.append(Integer(bound_min, bound_max, name=dim, dtype="int"))
        if isinstance(dimensions_dict[dim][0], float):
            dimensions.append(Real(bound_min, bound_max, name=dim))
    return dimensions
