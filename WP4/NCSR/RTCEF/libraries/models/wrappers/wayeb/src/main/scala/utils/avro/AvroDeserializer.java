package utils.avro;

import org.apache.avro.Schema;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.io.DatumReader;
import org.apache.avro.io.Decoder;
import org.apache.avro.io.DecoderFactory;
import org.apache.avro.specific.SpecificDatumReader;
import org.apache.kafka.common.serialization.Deserializer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.Scanner;


public class AvroDeserializer implements Deserializer<GenericRecord>{
    private DatumReader<GenericRecord> datumReader;


    @Override
    public void configure(Map<String, ?> configs, boolean isKey) {
        String type = isKey ? "key" : "value";
        String schemaPath = (String) configs.get(type+".avro.schema.path");

        var schema = Utils.parseSchemaFromPath(schemaPath);

        datumReader = new SpecificDatumReader<GenericRecord>(schema);

    }

    @Override
    public GenericRecord deserialize(String topic, byte[] data) {
        try {
            if (data == null)
                return null;
            else{
                Decoder decoder = DecoderFactory.get().binaryDecoder(data,null);
                return datumReader.read(null, decoder);
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
