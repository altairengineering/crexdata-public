try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json

import services.service as service


class Reader(service.Service):
    type = "reader"
    dataset_path = None
    filenames = []

    def __init__(self, configuration_file):
        super().__init__(configuration_file)
        path = self.service_parameters["outputfolder"]
        for input_topic_key in self.input_topic_var_keys:
            self.filenames.append((input_topic_key, f"{path}reader_{self.input_topic_var_keys[input_topic_key]}.txt"))

    def run(self):
        for filename in self.filenames:
            with open(filename[1], 'a') as f:
                records = self.consume(filename[0])
                for record in records:
                    print(record)
                    print(json.dumps(record), file=f)
