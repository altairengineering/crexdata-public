package stream.domain.maritime
import play.api.libs.json.{JsObject, Json}

import com.typesafe.scalalogging.LazyLogging

import stream.array.EventStream
import stream.source.EmitMode.EmitMode
import stream.source.{EmitMode, StreamSource}
import stream.{GenericEvent, ResetEvent}

object MaritimeWAStreamSourceJSON {
  /**
    * Constructor for maritime stream source.
    *
    * @param filename The path to the file.
    * @return The stream source.
    */
  def apply(filename: String): MaritimeWAStreamSourceJSON = new MaritimeWAStreamSourceJSON(filename)
}

/**
  * Stream source for maritime trajectories given in a single csv file.
  *
  * @param filename The path to the file.
  */
class MaritimeWAStreamSourceJSON(filename: String) extends StreamSource with LazyLogging {

  override protected def emitEvents(mode: EmitMode): EventStream = {
    val bufferedSource = io.Source.fromFile(filename)
    var totalCounter = 0L
    val eventStream = new EventStream()
    for (line <- bufferedSource.getLines) {
      val map = Json.parse(line).as[JsObject].value.toMap
      val timestamp = map.getOrElse("timestamp", totalCounter).toString.toLong
      val mmsi = map("mmsi").toString.filterNot("\"".toSet)
      val lon = map("lon").toString.toDouble
      val lat = map("lat").toString.toDouble
      val speed = map("speed").toString.toDouble
      val heading = map("trh").toString.toDouble
      val cog = map("cog").toString.toDouble
      val entryNearcoast = map("entry_nearcoast").toString.toDouble
      val entryNearcoast5k = map("entry_nearcoast5k").toString.toDouble
      val entryFishing = map("entry_fishing").toString.toDouble
      val entryNatura = map("entry_natura").toString.toDouble
      val entryNearports = map("entry_nearports").toString.toDouble
      val entryAnchorage = map("entry_anchorage").toString.toDouble
      val exitNearcoast = map("exit_nearcoast").toString.toDouble
      val exitNearcoast5k = map("exit_nearcoast5k").toString.toDouble
      val exitFishing = map("exit_fishing").toString.toDouble
      val exitNatura = map("exit_natura").toString.toDouble
      val exitNearports = map("exit_nearports").toString.toDouble
      val exitAnchorage = map("exit_anchorage").toString.toDouble

      val annotation = map("critical_bitstring").toString.filterNot("\"".toSet)
      val nextCETimestamp = map("next_timestamp").toString.toLong
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
      val arguments = argumentsA ++ expandedAnnotation
      if (timestamp == -1) send2Listeners(ResetEvent(Map("mmsi" -> mmsi)))
      else {
        send2Listeners(GenericEvent(totalCounter.toInt, "SampledCritical", timestamp, arguments))
      }
      totalCounter += 1

    }
    eventStream
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
}

