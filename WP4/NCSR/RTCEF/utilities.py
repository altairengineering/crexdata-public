import json

import fastavro
import io


def json_deserializer(m):
    """
    Json Deserializer. Deserializes ascii messsages that have a json format.

    :param m: an ascii string
    :return: a json dict
    """
    return json.loads(m.decode('ascii'))


def json_serializer(m):
    """
    Json serializer. Accepts a json (dict) and returns its serialized version
    :param m:
    :return: an ascii string
    """
    return json.dumps(m).encode('ascii')


def load_avro_schema_from_file(path):
    schema = fastavro.schema.load_schema(path)
    return schema


def avro_serializer(schema_path):
    schema_local = load_avro_schema_from_file(schema_path)

    def serializer(m):
        fo = io.BytesIO()
        fastavro.schemaless_writer(fo, schema_local, m)
        return fo.getvalue()

    return serializer


def avro_deserializer(schema_path):
    schema_local = load_avro_schema_from_file(schema_path)

    def deserializer(m):
        fo = io.BytesIO(m)
        return fastavro.schemaless_reader(fo, schema_local)

    return deserializer