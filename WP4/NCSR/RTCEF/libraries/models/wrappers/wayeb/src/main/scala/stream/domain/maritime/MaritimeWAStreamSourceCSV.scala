package stream.domain.maritime

import com.typesafe.scalalogging.LazyLogging
import stream.source.CSVStreamSource
import stream.{GenericEvent, ResetEvent}

object MaritimeWAStreamSourceCSV {
  /**
    * Constructor for maritime stream source.
    *
    * @param filename The path to the file.
    * @return The stream source.
    */
  def apply(filename: String): MaritimeWAStreamSourceCSV = new MaritimeWAStreamSourceCSV(filename)
}

/**
  * Stream source for maritime trajectories given in a single csv file.
  *
  * @param filename The path to the file.
  */
class MaritimeWAStreamSourceCSV(filename: String) extends CSVStreamSource(filename) with LazyLogging {

  override def line2Event(
                           line: Seq[String],
                           id: Int
                         ): GenericEvent = {
    try {
      val timestamp = line.head.toLong
      val mmsi = line(1)
      val lon = line(2).toDouble
      val lat = line(3).toDouble
      val speed = line(4).toDouble
      val heading = line(5).toDouble
      val cog = line(6).toDouble
      val entryNearcoast = line(7).toDouble
      val entryNearcoast5k = line(8).toDouble
      val entryFishing = line(9).toDouble
      val entryNatura = line(10).toDouble
      val entryNearports = line(11).toDouble
      val entryAnchorage = line(12).toDouble
      val exitNearcoast = line(13).toDouble
      val exitNearcoast5k = line(14).toDouble
      val exitFishing = line(15).toDouble
      val exitNatura = line(16).toDouble
      val exitNearports = line(17).toDouble
      val exitAnchorage = line(18).toDouble

      val annotation = line(19)
      val nextCETimestamp = line(20).toLong
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
      if (timestamp == -1) ResetEvent(Map("mmsi" -> mmsi))
      else {
        GenericEvent(id, "SampledCritical", timestamp, arguments)
      }
    } catch {
      case e: Exception =>
        logger.warn("COULD NOT PARSE LINE " + line)
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
}

