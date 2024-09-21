package stream.domain.cards

import com.typesafe.scalalogging.LazyLogging

import com.typesafe.scalalogging.LazyLogging
import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.apache.kafka.clients.consumer.KafkaConsumer
import org.apache.kafka.clients.producer.KafkaProducer
import stream.source.OOFStreamSource
import stream.{GenericEvent, ResetEvent}
import utils.avro.{AvroConsumer, AvroProducer}

object TrxStreamSourceOOF {

  def apply(streamConsumer: AvroConsumer,
            modelConsumer: AvroConsumer,
            syncConsumer: AvroConsumer,
            syncProducer: AvroProducer,
           ): TrxStreamSourceOOF = new TrxStreamSourceOOF(
    streamConsumer,
    modelConsumer,
    syncConsumer,
    syncProducer,
    0
  )

  def apply(streamConsumer: AvroConsumer,
            modelConsumer: AvroConsumer,
            syncConsumer: AvroConsumer,
            syncProducer: AvroProducer,
            startTime: Long
           ): TrxStreamSourceOOF = new TrxStreamSourceOOF(
    streamConsumer,
    modelConsumer,
    syncConsumer,
    syncProducer,
    startTime
  )
}

class TrxStreamSourceOOF(streamConsumer: AvroConsumer,
                                modelConsumer: AvroConsumer,
                                syncConsumer: AvroConsumer,
                                syncProducer: AvroProducer,
                                startTime: Long
                               ) extends OOFStreamSource(streamConsumer, modelConsumer, syncConsumer, syncProducer, startTime) with LazyLogging {

  override def GenericRecord2Event(
                                    record: GenericRecord,
                                    id: Int
                                  ): GenericEvent = {
    try {
      val timestamp = record.get("timestamp").toString.toLong
      val trxno = record.get("trx_no").toString.toLong
      val pan = record.get("pan").toString
      val amount = record.get("amount").toString.toDouble
      val amountDiff = record.get("amount_diff").toString.toDouble
      val timeDiff = record.get("time_diff").toString.toLong
      val isFraud = record.get("is_fraud").toString.toLong
      val fraudType = record.get("fraud_type").toString.toLong

      val arguments = Map(
        "trxno" -> trxno,
        "time" -> timestamp,
        "pan" -> pan,
        "amount" -> amount,
        "amountDiff" -> amountDiff,
        "timeDiff" -> timeDiff,
        "isFraud" -> isFraud,
        "fraudType" -> fraudType,
        "nextCETimestamp" -> -1
      )
      GenericEvent(id, "Trx", timestamp, arguments)
    }
      catch {
      case e: Exception =>
        logger.warn("COULD NOT PROCESS RECORD " + record)
        throw new Error
    }
  }

  override def GetTimestamp(record: GenericRecord): Long = {
    record.get("timestamp").toString.toLong
  }

}

