package stream.domain.maritime

import com.typesafe.scalalogging.LazyLogging
import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.apache.kafka.clients.consumer.KafkaConsumer
import org.apache.kafka.clients.producer.KafkaProducer
import stream.source.OOFStreamSource
import stream.{GenericEvent, ResetEvent}
import utils.avro.{AvroConsumer, AvroProducer}

import java.util.Properties

object MaritimeWAStreamSourceOOF {
  /**
   * Constructor for maritime stream source.
   *
   * @param filename The path to the file.
   * @return The stream source.
   */
  def apply(streamConsumer: AvroConsumer,
            modelConsumer: AvroConsumer,
            syncConsumer: AvroConsumer,
            syncProducer: AvroProducer,
           ): MaritimeWAStreamSourceOOF = new MaritimeWAStreamSourceOOF(
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
           ): MaritimeWAStreamSourceOOF = new MaritimeWAStreamSourceOOF(
    streamConsumer,
    modelConsumer,
    syncConsumer,
    syncProducer,
    startTime
  )
}

/**
 * Stream source for maritime trajectories given in a single csv file.
 *
 * @param filename The path to the file.
 */
class MaritimeWAStreamSourceOOF(streamConsumer: AvroConsumer,
                                modelConsumer: AvroConsumer,
                                syncConsumer: AvroConsumer,
                                syncProducer: AvroProducer,
                                startTime: Long
                               ) extends OOFStreamSource(streamConsumer, modelConsumer, syncConsumer, syncProducer, startTime) with LazyLogging {

  private val forecastDistance = 600
  private val vesselTimestamps = collection.mutable.Map[String, Long]()

  override def GenericRecord2Event(
                                    record: GenericRecord,
                                    id: Int
                                  ): GenericEvent = {
    try {
      val timestamp = record.get("timestamp").toString.toLong
      val mmsi = record.get("mmsi").toString
      val lon = record.get("lon").toString.toDouble
      val lat = record.get("lat").toString.toDouble
      val speed = record.get("speed").toString.toDouble
      val heading = record.get("trh").toString.toDouble
      val cog = record.get("cog").toString.toDouble
      val entryNearcoast = record.get("entry_nearcoast").toString.toDouble
      val entryNearcoast5k = record.get("entry_nearcoast5k").toString.toDouble
      val entryFishing = record.get("entry_fishing").toString.toDouble
      val entryNatura = record.get("entry_natura").toString.toDouble
      val entryNearports = record.get("entry_nearports").toString.toDouble
      val entryAnchorage = record.get("entry_anchorage").toString.toDouble
      val exitNearcoast = record.get("exit_nearcoast").toString.toDouble
      val exitNearcoast5k = record.get("exit_nearcoast5k").toString.toDouble
      val exitFishing = record.get("exit_fishing").toString.toDouble
      val exitNatura = record.get("exit_natura").toString.toDouble
      val exitNearports = record.get("exit_nearports").toString.toDouble
      val exitAnchorage = record.get("exit_anchorage").toString.toDouble

      val annotation = record.get("critical_bitstring").toString
      val nextCETimestamp = record.get("next_timestamp").toString.toLong
      val gap_start = timestamp match {
        case -1 => 1
        case _ => 0
      }
      val expandedAnnotation = annotationExpansion(annotation)
      val argumentsA = Map("mmsi" -> mmsi, "speed" -> speed,
        "lon" -> lon, "lat" -> lat,
        "heading" -> heading, "cog" -> cog,
        "entryNearcoast" -> entryNearcoast,
        "entryNearcoast5k" -> entryNearcoast5k,
        "entryFishing" -> entryFishing,
        "entryNatura" -> entryNatura,
        "entryNearports" -> entryNearports,
        "entryAnchorage" -> entryAnchorage,
        "exitNearcoast" -> exitNearcoast,
        "exitNearcoast5k" -> exitNearcoast5k,
        "exitFishing" -> exitFishing,
        "exitNatura" -> exitNatura,
        "exitNearports" -> exitNearports,
        "exitAnchorage" -> exitAnchorage,
        "nextCETimestamp" -> nextCETimestamp,
        "gap_start" -> gap_start
      )

      val isEmitting = collection.mutable.Map("isEmitting" -> "false")
      if (vesselTimestamps.contains(mmsi)){
        val timeToEmit = vesselTimestamps(mmsi)
        if (timestamp >= timeToEmit){
          isEmitting("isEmitting") = "true"
          vesselTimestamps(mmsi) = timeToEmit + forecastDistance
        }
      } else {
        vesselTimestamps(mmsi) = timestamp + forecastDistance
      }
      val arguments = argumentsA ++ expandedAnnotation ++ isEmitting

      if (timestamp == -1){
        if (vesselTimestamps.contains(mmsi)) vesselTimestamps.remove(mmsi)
        ResetEvent(Map("mmsi" -> mmsi))
      }
      else {
        GenericEvent(id, "SampledCritical", timestamp, arguments)
      }
    } catch {
      case e: Exception =>
        logger.warn("COULD NOT PROCESS RECORD " + record)
        throw new Error
    }
  }

  private def annotationExpansion(
                                   criticalAnnotationString: String
                                 ): Map[String, Double] =
    criticalAnnotationString match {
      case "-1" => Map(
        "stop_start" -> -1.0,
        "stop_end" -> -1.0,
        "slow_motion_start" -> -1.0,
        "slow_motion_end" -> -1.0,
        "gap_end" -> -1.0,
        "change_in_heading" -> -1.0,
        "change_in_speed_start" -> -1.0,
        "change_in_speed_end" -> -1.0
      )
      case _ => Map(
        "stop_start" -> criticalAnnotationString.slice(7, 8).toDouble,
        "stop_end" -> criticalAnnotationString.slice(6, 7).toDouble,
        "slow_motion_start" -> criticalAnnotationString.slice(5, 6).toDouble,
        "slow_motion_end" -> criticalAnnotationString.slice(4, 5).toDouble,
        "gap_end" -> criticalAnnotationString.slice(3, 4).toDouble,
        "change_in_heading" -> criticalAnnotationString.slice(2, 3).toDouble,
        "change_in_speed_start" -> criticalAnnotationString.slice(1, 2).toDouble,
        "change_in_speed_end" -> criticalAnnotationString.slice(0, 1).toDouble
      )
    }

  override def GetTimestamp(record: GenericRecord): Long = {
    record.get("timestamp").toString.toLong
  }

}

