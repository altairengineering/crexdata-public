package stream.source

import model.waitingTime.ForecastMethod.ForecastMethod
import org.apache.avro.Schema
import org.apache.avro.generic.{GenericData, GenericRecord}
import stream.array.EventStream
import stream.source.UpdateModelEvent
import stream.source.EmitMode.EmitMode
import org.apache.kafka.clients.consumer.{ConsumerRecords, KafkaConsumer}
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import stream.GenericEvent
import utils.avro.{AvroConsumer, AvroProducer}
import play.api.libs.json.{JsObject, Json}

import java.time.Duration
import scala.jdk.CollectionConverters.iterableAsScalaIterableConverter


/**
 * A stream source for the online optimization framework.
 * @param streamConsumer      a KafkaConsumer object that consumes the input stream
 * @param modelConsumer       a KafkaConsumer object that consumes model record
 * @param syncConsumer        a KafkaConsumer object that consumes sync records
 * @param syncProducer        a KafkaProducer object that should write to the sync topic that the collector reads
 * */
abstract class OOFStreamSource(streamConsumer: AvroConsumer,
                               modelConsumer: AvroConsumer,
                               syncConsumer : AvroConsumer,
                               syncProducer : AvroProducer,
                               startTime: Long
                       ) extends StreamSource {

  /**
    * For BUFFER mode, simply return the array. For ONLINE, send all events to listeners.
    *
    * @param mode The mode, BUFFER or ONLINE.
    * @return The stream as an array of events.
    */
  override  def emitEvents(mode: EmitMode): EventStream = {
    val eventStream = new EventStream()

    var paused:Boolean = false
    var transition:Boolean = false
    var newModel:Boolean = false
    var newModelInfo:GenericRecord = null
    var totalCounter = 1
    var updateCounter = 1
    var currentTime:Long = 0
    var syncTime:Long = 0
    var currentOffset:Long = 0
    val duration = 100
    // online emit mode is supported atm
    assert(mode == EmitMode.ONLINE)

    while(true){

      val syncRecords: ConsumerRecords[GenericRecord, GenericRecord] = syncConsumer.consume(duration)
      // Syncing:
      // if pause is received
      // wait for "new model" and "play" messages
      // if play received and new model received then:
      //   consume up to current_timestamp then
      //   update model

      for (syncRecord <- syncRecords.asScala){
        val typeOfMsg = syncRecord.value().get("type").toString()
        println(typeOfMsg)
        typeOfMsg match {
          case "pause" =>
            paused = true
          case "play" =>
            transition = true
        }
      }
      if (transition) {
        val modelRecords: ConsumerRecords[GenericRecord, GenericRecord] = modelConsumer.consume(duration)
        // if multiple models get last? yes for now
        for (modelRecord <- modelRecords.asScala) {
          newModelInfo = modelRecord.value()
          val productionTime = newModelInfo.get("pt").asInstanceOf[Long]
          syncTime = currentTime + productionTime
          newModel = true
          paused = false
        }
      }
      if(!paused){
        val streamRecords: ConsumerRecords[GenericRecord, GenericRecord] = streamConsumer.consume(duration)
        for (streamRecord <- streamRecords.asScala) {
          currentTime = GetTimestamp(streamRecord.value())
          currentOffset = streamRecord.offset()
          // sync with collect TODO partition per offset
          SyncWithCollector(currentOffset, currentTime)
          if (currentTime > startTime) {
            totalCounter += 1

            if (transition) {
              // in here we are in the process of changing the model
              // we should consume records until currentime >= synctime
              // and then update  the model
              if (currentTime < syncTime) {
                // consume records with old model
                //send event to listener
                val event = GenericRecord2Event(streamRecord.value(), totalCounter)
                send2Listeners(event)
              } else {
                //update model
                //create arguments for UpdateModelEvent
                val modelPath = newModelInfo.get("path").toString
                val confidenceThreshold = Json.parse(newModelInfo.get("model_params").toString).
                  as[JsObject].value.toMap.get("confidence").get.toString

                val args = Map(
                  "Path" -> modelPath,
                  "confidenceThreshold" -> confidenceThreshold,
                  "id" -> newModelInfo.get("id").toString,
                  "creationMethod" -> newModelInfo.get("creation_method").toString
                )
                val event = new UpdateModelEvent(updateCounter, currentTime, args)
                updateCounter += 1
                transition = false
                send2Listeners(event)
              }
            } else {
              // normal consumption
              val event = GenericRecord2Event(streamRecord.value(), totalCounter)
              send2Listeners(event)
            }
          }
        }
      }
    }
    eventStream
  }

  private def SyncWithCollector(currentOffset:Long, currentTime:Long): Unit = {
    val syncKeyRecord = new GenericData.Record(syncProducer.schemas("key"))
    val syncValueRecord = new GenericData.Record(syncProducer.schemas("value"))
    syncKeyRecord.put("offset", currentOffset)
    syncKeyRecord.put("timestamp", currentTime)
    syncValueRecord.put("offset", currentOffset)
    syncValueRecord.put("timestamp", currentTime)
    syncProducer.send(syncKeyRecord, syncValueRecord)
  }
  protected def GenericRecord2Event(
                            record: GenericRecord,
                            id: Int
                          ): GenericEvent

  protected def GetTimestamp(record: GenericRecord): Long
}
