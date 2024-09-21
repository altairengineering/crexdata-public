from libraries.models.wrappers import wayeb_wraper


class Forecaster:
    model = None

    def __init__(self, model):
        if model == "wayeb":
            # model = wayeb_wraper.WayebWrapper(gate_run, k_val, weights, threshold_time)
            pass

    def train_and_save(self, **params):
        pass

    def load_and_test(self, **params):
        pass

    def stream_process(self):
        pass
