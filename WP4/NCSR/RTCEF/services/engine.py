import services.service as service
from libraries.models import utils


class Engine(service.Service):
    type = "engine"
    # input output stream
    input_stream = "inputstream"
    output_stream = "outputstream"
    # syncing
    engine_position = "engineposition"
    engine_sync = "enginesync"
    # services
    model_versions = "models"
    score_reports = "scorereports"

    current_records_number = 0
    i = 0

    def __init__(self, configuration_file, model="wayeb"):
        super().__init__(configuration_file)
        self.model = model
        self.send_offset = int(self.service_parameters["updatePosition"])

    def run(self):
        # call wayeb
        if self.model == "wayeb":
            wayeb_jar = self.service_parameters["model.wayebjar"]
            java_port = int(self.service_parameters["model.param.java_port"])
            python_port = int(self.service_parameters["model.param.python_port"])
            initial_model = self.service_parameters["model.param.initial_model"]
            confidence_threshold = float(self.service_parameters["model.param.confidence_threshold"])
            start_time = int(self.service_parameters["model.param.start_time"])
            min_distance = float(self.service_parameters["model.param.distance_min"])
            max_distance = float(self.service_parameters["model.param.distance_max"])
            domain = self.service_parameters["model.param.domain"]

            schemas_path = self.service_parameters["schemaspath"]
            stats_reporting_distance = int(self.service_parameters["model.param.reporting_dt"])

            gateway = utils.build_gateway(python_port, java_port)
            run = gateway.entry_point
            print("Starting wayeb engine...", flush=True)
            run.runOOFEngine(initial_model,
                             min_distance,
                             max_distance,
                             confidence_threshold,
                             self.input_topic_var_keys[self.input_stream],
                             self.input_topic_var_keys[self.engine_sync],
                             self.input_topic_var_keys[self.model_versions],
                             self.output_topic_var_keys[self.engine_position],
                             self.output_topic_var_keys[self.score_reports],
                             self.output_topic_var_keys[self.output_stream],
                             schemas_path,
                             stats_reporting_distance,
                             start_time,
                             domain
                             )
            print("Hmmmm...", flush=true)

    pass
