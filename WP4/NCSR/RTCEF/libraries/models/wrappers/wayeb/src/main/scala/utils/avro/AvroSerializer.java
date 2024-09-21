package utils.avro;

import org.apache.avro.Schema;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.io.DatumWriter;
import org.apache.avro.io.Encoder;
import org.apache.avro.io.EncoderFactory;
import org.apache.avro.specific.SpecificDatumWriter;
import org.apache.kafka.common.serialization.Serializer;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.Scanner;

public class AvroSerializer implements Serializer<GenericRecord> {

    private DatumWriter<GenericRecord> datumWriter;

    @Override
    public void configure(Map configs, boolean isKey) {
        String type = isKey ? "key" : "value";
        String schemaPath = (String) configs.get(type+".avro.schema.path");

        var schema = Utils.parseSchemaFromPath(schemaPath);
        datumWriter = new SpecificDatumWriter<GenericRecord>(schema);
    }


    @Override
    public byte[] serialize(String topic, GenericRecord data) {

        try {
            if (data == null)
                return null;
            else {
                ByteArrayOutputStream out = new ByteArrayOutputStream();
                Encoder encoder = EncoderFactory.get().binaryEncoder(out,null);
                datumWriter.write(data, encoder);
                encoder.flush();
                out.close();
                return out.toByteArray();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
