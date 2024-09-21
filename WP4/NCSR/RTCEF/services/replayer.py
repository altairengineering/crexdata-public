from collections import deque

try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json

import services.service as service


class Replayer(service.Service):
    type = "replayer"
    input_stream = "inputstream"
    dataset_path = None
    cyclic = False
    period_limits = []
    time = "sec"
    domain = "maritime"

    def __init__(self, configuration_file):
        super().__init__(configuration_file)
        self.dataset_path = self.service_parameters["datasetpath"]
        self.start_period = int(self.service_parameters["startperiod"])
        self.start_time = int(self.service_parameters["starttime"])
        self.end_time = int(self.service_parameters["endtime"])
        self.cyclic = self.service_parameters["cyclic"] == "true"
        if "time" in self.service_parameters:
            self.time = self.service_parameters["time"]

        if self.time == "sec":
            multiplier = 1
        else:
            multiplier = 1000

        self.period_size = int(self.service_parameters["periodsize"]) * 86400 * multiplier

        if "domain" in self.service_parameters:
            self.domain = self.service_parameters["domain"]

        start_time = self.start_time
        self.period_limits = []

        while start_time <= self.end_time:
            self.period_limits.append([start_time, start_time + self.period_size])
            start_time = start_time + self.period_size

        print(f"Period limits: {self.period_limits}")

    def run(self):
        with open(self.dataset_path, "r") as inp:
            i = 0
            if not self.cyclic:
                for line in inp:
                    json_record = json.loads(line)
                    self.domain_produce(self.input_stream, [json_record])

                    i = i + 1
                    if self.interrupted:
                        break
            if self.cyclic:
                print(f"Cyclic mode on - Start period {self.start_period}")
                if self.start_period == 0:
                    with open(self.dataset_path, "r") as inp:
                        i = 0
                        for line in inp:
                            json_record = json.loads(line)
                            self.domain_produce(self.input_stream, [json_record])

                            i = i + 1
                            if self.interrupted:
                                break
                else:
                    offset = -1
                    last_timestamp = -1
                    i = 0
                    with open(self.dataset_path, "r") as inp:
                        for line in inp:
                            json_record = json.loads(line)

                            if json_record["timestamp"] >= self.period_limits[self.start_period][0]:
                                if offset < 0:
                                    offset = json_record["timestamp"] - self.start_time
                                # print(f"{json_record['timestamp']}, {json_record['timestamp'] - offset}")
                                json_record["timestamp"] = json_record["timestamp"] - offset
                                last_timestamp = json_record["timestamp"]
                                self.domain_produce(self.input_stream, [json_record])
                                i = i + 1

                            if self.interrupted:
                                break
                    print("Changing....")
                    offset = -1
                    with open(self.dataset_path, "r") as inp:
                        for line in inp:
                            json_record = json.loads(line)

                            if json_record["timestamp"] >= self.period_limits[self.start_period][0]:
                                break

                            if offset < 0:
                                offset = last_timestamp - json_record["timestamp"]
                            # print(f"{json_record['timestamp']}, {json_record['timestamp'] - offset}")
                            json_record["timestamp"] = json_record["timestamp"] + offset
                            self.domain_produce(self.input_stream, [json_record])

                            if self.interrupted:
                                break

    def domain_produce(self, topic, record_list):
        if self.domain == "maritime":
            self.produce(topic, record_list, ["timestamp", "mmsi"], noprints=True)
        if self.domain == "cards":
            self.produce(topic, record_list, ["trx_no"], noprints=True)

def head(filename):
    """Return first line of file"""
    with open(filename) as f:
        return f.readline()


def tail(filename):
    """Return the last n lines of a file"""
    with open(filename) as f:
        return deque(f, 1)[0]
