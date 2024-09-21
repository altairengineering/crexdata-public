import configparser
import uuid

import utilities
import re


def string_is_float(string):
    if re.match(r'^-?\d+(?:\.\d+)$', string) is None:
        return False
    else:
        return True


def string_is_integer(string):
    if re.match("[-+]?\d+$", string) is None:
        return False
    else:
        return True


def get_service_parameters(config, service_name):
    """
    Given a config object, and a service name return the parameters of that service.

    :param config: config object
    :param service_name: service name
    :return: config section
    """
    if service_name not in config:
        raise Exception(f"Configuration parsing error:",
                        f"section {service_name} not found in provided config.")
    return config[service_name]


def get_input_output_topics_from_config(config, service_name):
    """
    Accepts a config object and returns
    :param service_name: name of the service
    :param config: a configuration
    :return: Two dicts. First list contains var names (key) and actual names (value)
    of input topics, second list contains same data for output topics.
    """
    topics_section_name = service_name + ".topics"
    if topics_section_name not in config:
        raise Exception(f"Configuration parsing error:",
                        f"section {topics_section_name} not found in provided config.")
    topics_section = config[topics_section_name]
    input_topic_keys = {}
    output_topic_keys = {}

    for topic in topics_section:
        # Input.varname = topicname
        # Output.varname = topicname
        topic_type_var = topic.split(".")
        topic_type = topic_type_var[0]
        topic_var = topic_type_var[1]
        if topic_type == "input":
            input_topic_keys[topic_var] = topics_section[topic]
        elif topic_type == "output":
            output_topic_keys[topic_var] = topics_section[topic]

    schemas_section_name = topics_section_name+".schemas"
    input_topic_schemas = {}
    output_topic_schemas = {}

    if schemas_section_name in config:
        schemas_section = config[schemas_section_name]
        for topic_var in input_topic_keys:
            if "input."+topic_var in schemas_section:
                input_topic_schemas[topic_var] = schemas_section["input."+topic_var]
        for topic_var in output_topic_keys:
            if "output."+topic_var in schemas_section:
                output_topic_schemas[topic_var] = schemas_section["output."+topic_var]

    return input_topic_keys, output_topic_keys, input_topic_schemas, output_topic_schemas


def parse_configuration(configuration_file_path):
    """
    Returns class configuration

    :param configuration_file_path: path to the configuration file
    :return: a config object
    """
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(configuration_file_path)
    return config


def update_kafka_settings(settings, kafka_settings, process_type, schema_path=None):
    """
    Updates dict "settings" with kafka settings provided in configuration
    "kafka_settings". Process type should be either consumer or producer.

    :param settings: settings dict to get updated
    :param kafka_settings: configuration that contains kafka settings
    :param process_type: consumer/producer
    """
    custom_settings = {}
    for var in kafka_settings:
        var_types = var.split(".")
        if var_types[0] == "custom" and (var_types[1] == process_type or var_types[1] == "all"):
            custom_settings[var_types[2]] = kafka_settings[var]
        elif var_types[0] == process_type or var_types[0] == "all":
            if string_is_float(kafka_settings[var]):
                settings[var_types[1]] = float(kafka_settings[var])
            elif string_is_integer(kafka_settings[var]):
                settings[var_types[1]] = int(kafka_settings[var])
            else:
                settings[var_types[1]] = kafka_settings[var]

    key_deserializer = None
    value_deserializer = None
    key_serializer = None
    value_serializer = None

    for var in custom_settings:
        if process_type == "consumer":
            if var == "key_deserializer" and custom_settings[var] == "json":
                key_deserializer = utilities.json_deserializer
            if var == "key_deserializer" and custom_settings[var] == "avro":
                key_deserializer = utilities.avro_deserializer(schema_path+"-key.avsc")
            if var == "value_deserializer" and custom_settings[var] == "json":
                value_deserializer = utilities.json_deserializer
            if var == "value_deserializer" and custom_settings[var] == "avro":
                value_deserializer = utilities.avro_deserializer(schema_path+"-value.avsc")
        else:
            if var == "key_serializer" and custom_settings[var] == "json":
                key_serializer = utilities.json_serializer
            if var == "key_serializer" and custom_settings[var] == "avro":
                key_serializer = utilities.avro_serializer(schema_path+"-key.avsc")
            if var == "value_serializer" and custom_settings[var] == "json":
                value_serializer = utilities.json_serializer
            if var == "value_serializer" and custom_settings[var] == "avro":
                value_serializer = utilities.avro_serializer(schema_path+"-value.avsc")

    if process_type == "consumer":
        settings["key_deserializer"] = key_deserializer
        settings["value_deserializer"] = value_deserializer
    elif process_type == "producer":
        settings["key_serializer"] = key_serializer
        settings["value_serializer"] = value_serializer


def get_configuration(configuration, process_type, service_name, schema_path=None):
    kafka_settings = None
    if process_type not in ("consumer", "producer"):
        raise Exception("Configuration parsing error:", f"Asked type '{process_type}' is not in available options.")

    settings = {}

    if "kafka" not in configuration:
        raise Exception("Configuration parsing error:", "kafka section not found.")
    else:
        kafka_settings = configuration["kafka"]
        update_kafka_settings(settings, kafka_settings, process_type, schema_path)

    if service_name + ".kafka" in configuration:
        service_kafka_settings = configuration[service_name + ".kafka"]
        update_kafka_settings(settings, service_kafka_settings, process_type, schema_path)
    if process_type == "consumer":
        settings["group_id"] = service_name+"_"+str(uuid.uuid4())
    if len(settings) == 0:
        raise Exception("Configuration parsing error:", "kafka settings not found.")

    return settings


def get_parameters_domain(service_parameters):
    parameters_domain = {}
    for var in service_parameters:

        split = var.split(".")
        if split[0] == "hyper":
            param_name = split[1]
            if param_name not in parameters_domain:
                parameters_domain[param_name] = [None, None]

            bound = split[2]
            value = service_parameters[var]
            if "." in value:
                value = float(value)
            else:
                value = int(value)
            if bound == "min":
                parameters_domain[param_name][0] = value
            elif bound == "max":
                parameters_domain[param_name][1] = value
    return parameters_domain


def get_model_fixed_params(service_parameters):
    fixed_params = {}
    for var in service_parameters:
        if "model.param" in var:
            param_name = var.split(".")[-1]
            value = service_parameters[var]
            if string_is_float(value):
                value = float(value)
            elif string_is_integer(value):
                value = int(value)
            fixed_params[param_name] = value
    return fixed_params
