package stream.domain.archived

import com.typesafe.scalalogging.LazyLogging
import stream.GenericEvent
import stream.array.{EventStream, EventStreamI}
//import streamer.FStreamer

object MaritimeStream {
  def apply(filename: String, sampleInterval: Int): MaritimeStream = new MaritimeStream(filename, sampleInterval)

  def apply(filename: String): MaritimeStream = new MaritimeStream(filename, -1)
}

class MaritimeStream(
    filename: String,
    sampleInterval: Int
) extends EventStreamI with LazyLogging {
  override def generateStream(): EventStream = {
    val eventStream = new EventStream()
    var totalCounter = 1
    var previousTimestamp: Long = -1
    val bufferedSource = io.Source.fromFile(filename)
    for (line <- bufferedSource.getLines) {
      val ev = line2Event(line, totalCounter, previousTimestamp)
      totalCounter += 1
      eventStream.addEvent(ev.head)
      if (ev.length > 1) {
        eventStream.addEvent(ev.tail.head)
      }
      if (sampleInterval != -1) {
        previousTimestamp = ev.head.timestamp
      }
    }
    /*val mee = new MaritimeEventExtractor
    val mb = new MaritimeBuilder
    val mf = new java.io.File(filename)
    val fs = FStreamer[GenericEvent](mf,mee,mb,1)
    eventStream.setStreamer(fs)*/
    eventStream
  }

  private def line2Event(
      line: String,
      id: Int,
      previousTimestamp: Long
  ): List[GenericEvent] = {
    val cols = line.split(",").map(_.trim)
    try {
      val mmsi = s"${cols(0)}"
      val speed = s"${cols(1)}".toDouble
      val heading = s"${cols(2)}".toDouble
      val lon = s"${cols(3)}".toDouble
      val lat = s"${cols(4)}".toDouble
      val timestamp = s"${cols(5)}".toLong
      /*val within = s"${cols(6)}"
      val d57 = s"${cols(7)}"
      val d710 = s"${cols(8)}"
      val spatial = extractSpatial(within,d57,d710)*/
      val timeDiff = timestamp - previousTimestamp
      val ge = GenericEvent(id, "SampledCritical", timestamp,
                            Map("mmsi" -> mmsi, "speed" -> speed, "lon" -> lon, "lat" -> lat, "heading" -> heading)) //,
      //"within"->spatial("within"),"d57"->spatial("d57"),"d710"->spatial("d710")))
      if (previousTimestamp == -1 | timeDiff == sampleInterval) List(ge)
      else {
        val resetEvent = GenericEvent(id, "RESET", timestamp,
                                      Map("mmsi" -> mmsi, "speed" -> speed, "lon" -> lon, "lat" -> lat, "heading" -> heading))
        List(resetEvent, ge)
      }
    } catch {
      case e: Exception => {
        logger.warn("COULD NOT PARSE EVENT")
        List(GenericEvent(-1, "NA", -1))
      }

    }
  }

  private def extractSpatial(
      within: String,
      d57: String,
      d710: String
  ): Map[String, String] = {
    val withinPorts = within.split("\\.").toList.drop(1).mkString(",")
    val d57Ports = d57.split("\\.").toList.drop(1).mkString(",")
    val d710Ports = d710.split("\\.").toList.drop(1).mkString(",")
    Map("within" -> withinPorts, "d57" -> d57Ports, "d710" -> d710Ports)
  }

}
