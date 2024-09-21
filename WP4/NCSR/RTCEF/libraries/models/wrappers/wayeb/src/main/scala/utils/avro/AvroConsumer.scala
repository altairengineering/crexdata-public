package utils.avro

import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.apache.kafka.clients.consumer.{ConsumerRecords, KafkaConsumer}
import utils.avro

import java.time.Duration
import java.util.{Collections, Properties}

object AvroConsumer {
  def apply(properties: Properties, schemaPaths: Map[String, String], topic:String): AvroConsumer = {
      properties.put("value.avro.schema.path", schemaPaths("value"))
      properties.put("key.avro.schema.path", schemaPaths("key"))
      properties.put("key.deserializer", classOf[AvroDeserializer].getName)
      properties.put("value.deserializer", classOf[AvroDeserializer].getName)
      new AvroConsumer(properties: Properties, schemaPaths: Map[String, String], topic:String)
    }
}

class AvroConsumer(properties: Properties, schemaPaths: Map[String, String], topic:String) extends KafkaConsumer(properties) {
  private val keySchemaPath:String = schemaPaths("key")
  private val valueSchemaPath:String = schemaPaths("value")

  val schemas: Map[String, Schema] = Map(
    "key" -> avro.Utils.parseSchemaFromPath(keySchemaPath),
    "value" -> avro.Utils.parseSchemaFromPath(valueSchemaPath)
  )

  val consumer: KafkaConsumer[GenericRecord, GenericRecord] =
    new KafkaConsumer[GenericRecord, GenericRecord](properties)

  consumer.subscribe(Collections.singletonList(topic))

  def consume(durationMS: Int):ConsumerRecords[GenericRecord,GenericRecord] = {
    val duration = Duration.ofMillis(durationMS)
    consumer.poll(duration)
  }
}
