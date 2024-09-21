import kafka
import logging
from configuration import (parse_configuration,
                           get_input_output_topics_from_config,
                           get_configuration, get_service_parameters)

from kafka.structs import TopicPartition
import signal


class Service(object):
    type = "generic"

    consumers = {}
    producers = {}

    FORMAT = '%(asctime)s - %(message)s'
    log_filename = None
    logger = None
    interrupted = False
    config = None
    service_parameters = None

    def __init__(self, configuration_file):
        """

        Initialisation of the generic service class.

        :param: path to a configuration file
        """

        # assign signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

        # parse configuration
        self.config = parse_configuration(configuration_file)
        self.service_parameters = get_service_parameters(self.config, self.type)

        if "executionlog" in self.service_parameters:
            self.log_filename = self.service_parameters["executionlog"]

        # create logger
        logging.basicConfig(format=self.FORMAT, filename=self.log_filename, level=logging.INFO)
        self.logger = logging.getLogger(self.type)
        self.logger_extras = {}

        self.logger.info(f"Creating {self.type} service...")
        (self.input_topic_var_keys, self.output_topic_var_keys,
         self.input_topic_var_schemas, self.output_topic_var_schemas) = get_input_output_topics_from_config(self.config,
                                                                                                            self.type)
        configurations = {}

        if len(self.input_topic_var_keys) == 0 and len(self.output_topic_var_keys) == 0:
            self.logger.error(f"{self.type} error: Service has no consumers/producer.", extra=self.logger_extras)
            raise Exception(f"{self.type} error:", "Service has no consumers/producer.")

        if configurations is None:
            raise Exception(f"{self.type} error:", "Initialisation is impossible without configuration.")

        # create consumers
        for input_topic_var in self.input_topic_var_keys:
            self.logger.info(f"Creating consumers...")
            topic_name = self.input_topic_var_keys[input_topic_var]
            # get configuration
            if input_topic_var in self.input_topic_var_schemas:
                schema_path = self.input_topic_var_schemas[input_topic_var]
                configurations[input_topic_var] = get_configuration(self.config, "consumer",
                                                                    self.type, schema_path)
            # try to create a kafka consumer
            try:
                tp = TopicPartition(topic_name, 0)
                # self.consumers[input_topic_var] = kafka.KafkaConsumer(topic_name,
                #                                                       **configurations[input_topic_var])
                self.consumers[input_topic_var] = kafka.KafkaConsumer(**configurations[input_topic_var])
                self.consumers[input_topic_var].assign([tp])

                self.logger.info(f"Created Kafka Consumer for topic {topic_name}.", extra=self.logger_extras)
            except Exception:
                self.logger.error(
                    f"{self.type} error: Could not initialise KafkaConsumer for topic {topic_name}.",
                    extra=self.logger_extras)
                raise Exception(f"{self.type} error:",
                                f"Could not initialise KafkaConsumer for topic {topic_name}.")
        for output_topic_var in self.output_topic_var_keys:
            self.logger.info("Creating producers...")
            if output_topic_var in self.output_topic_var_schemas:
                schema_path = self.output_topic_var_schemas[output_topic_var]
                configurations[output_topic_var] = get_configuration(self.config,
                                                                     "producer", self.type, schema_path)
            else:
                configurations[output_topic_var] = get_configuration(self.config,
                                                                     "producer", self.type)
            try:
                self.producers[output_topic_var] = kafka.KafkaProducer(**configurations[output_topic_var])
                self.logger.info(f"Created Kafka Producer.", extra=self.logger_extras)
            except Exception:
                self.logger.error(f"{self.type} error: Could not initialise KafkaProducer.",
                                  extra=self.logger_extras)
                raise Exception(f"{self.type} error:",
                                f"Could not initialise KafkaProducer.")

    def consume(self, input_topic_key, max_records=1, add_offset=False, max_offset=None):
        """
        Consumption function of service. Calling this function will call poll
        on the specified consumer.

        :param add_offset: if true offset of msg will be added as field in record
        :param max_offset: consume records only up to max_offset
        :param input_topic_key: name of the topic
        :param max_records: the maximum number of records to consume
        :return: will consume max_records record(s) from subscribed topic(s)
        """
        if input_topic_key not in self.consumers:
            self.logger.error(f"Consumer of {self.type} service error: Key {input_topic_key} missing from consumers.",
                              extra=self.logger_extras)
            raise Exception(f"Consumer of {self.type} service error:", f"Key {input_topic_key} missing from consumers.")
        # print(f"Service {self.type}: Waiting for record...")
        messages = []
        if (max_offset is None) or max_offset >= self.consumers[input_topic_key].position(
                TopicPartition(self.input_topic_var_keys[input_topic_key], 0)):
            records = self.consumers[input_topic_key].poll(max_records=max_records)
            # print(f"Consumed record {record}")
            for topic_data, consumer_records in records.items():
                for consumer_record in consumer_records:
                    tmp_record = consumer_record.value
                    if add_offset:
                        tmp_record["offset"] = consumer_record.offset
                    messages.append(tmp_record)
        return messages

    def produce(self, topic_var, records, keys=None, noprints=False):
        """

        :param keys: record keys to use for key of message
        :param noprints: if false no print will be produced in logs.
        :param topic_var: var name of the topic to send records
        :param records: a list of records
        """
        if keys is None:
            keys = []
        topic = self.output_topic_var_keys[topic_var]
        if topic_var not in self.producers:
            self.logger.error(f"{self.type} service error: Called a producer for topic {topic_var},"
                              f" but no producer exists for that topic.",
                              extra=self.logger_extras)
            raise Exception(f"{self.type} service error:", f"Called a producer for topic {topic_var}"
                                                           f" but no producer exists for that topic.")
        for record in records:
            # try:
            if not noprints:
                self.logger.info(f"Service {self.type} sends a record to topic with name {topic}.",
                                 extra=self.logger_extras)
            record_key = {k: record[k] for k in keys if k in record}
            self.producers[topic_var].send(topic=topic, value=record, key=record_key)
            # except Exception:
            #     self.logger.error(f"{self.type} service error: Could not send record to topic {topic}.",
            #                       extra=self.logger_extras)
            #     raise Exception(f"{self.type} service error:", f"Could not send record to topic {topic}.")
        pass

    def signal_handler(self, signal, frame):
        self.interrupted = True

    def deploy(self, no_loop=False):
        """
        Deploy service. I.e., start run method in an infinite loop.
        """
        self.logger.info(f"Service {self.type} started running...")
        if not no_loop:
            while True:
                self.run()
                if self.interrupted:
                    print(f"Stopping service {self.type}...")
                    self.cleanup()
                    break
        else:
            self.run()
            print(f"Stopping service {self.type}...")
            self.cleanup()

    def cleanup(self):
        for consumer in self.consumers:
            self.consumers[consumer].close()
        pass

    def run(self):
        pass
