package utils.avro

import org.apache.avro.Schema
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import utils.avro

import java.util.Properties

object AvroProducer {
  def apply(properties: Properties, schemaPaths: Map[String, String], topic:String):AvroProducer = {
    properties.put("value.avro.schema.path", schemaPaths("value"))
    properties.put("key.avro.schema.path", schemaPaths("key"))
    properties.put("key.serializer",  classOf[AvroSerializer].getName)
    properties.put("value.serializer", classOf[AvroSerializer].getName)
    new AvroProducer(properties: Properties, schemaPaths: Map[String, String], topic:String)
  }
}

class AvroProducer(properties: Properties, schemaPaths: Map[String, String], topic:String) extends KafkaProducer(properties){
  private val keySchemaPath:String = schemaPaths("key")
  private val valueSchemaPath:String = schemaPaths("value")

  val schemas: Map[String, Schema] = Map(
    "key" -> avro.Utils.parseSchemaFromPath(keySchemaPath),
    "value" -> avro.Utils.parseSchemaFromPath(valueSchemaPath)
  )

  val producer: KafkaProducer[GenericRecord, GenericRecord] =
    new KafkaProducer[GenericRecord, GenericRecord](properties)

  def send(keyRecord:GenericData.Record, valueRecord:GenericData.Record):Unit = {
    val record = new ProducerRecord[GenericRecord, GenericRecord](topic, keyRecord, valueRecord)
    producer.send(record)
  }


}
