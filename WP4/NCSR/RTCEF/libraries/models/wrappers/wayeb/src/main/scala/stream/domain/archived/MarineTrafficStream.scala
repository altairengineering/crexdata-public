package stream.domain.archived

import java.time.{ZoneId, ZonedDateTime}

import stream.GenericEvent
import stream.array.{EventStream, EventStreamI}

object MarineTrafficStream {
  def apply(
      filename: String,
      shipIds: Set[String],
      start: String,
      end: String
  ): MarineTrafficStream = {
    val startUnix = indate2Unix(start)
    val endUnix = indate2Unix(end)
    new MarineTrafficStream(filename, shipIds, startUnix, endUnix)
  }

  def apply(filename: String) = new MarineTrafficStream(filename, Set.empty[String], -1, -1)

  def apply(
      filename: String,
      shipIds: Set[String]
  ) = new MarineTrafficStream(filename, shipIds, -1, -1)

  private def indate2Unix(str: String): Long = {
    val zoneId = ZoneId.of("UTC")

    val dateNtime = str.split("=").map(_.trim)
    val date = dateNtime(0)
    val time = dateNtime(1)
    val splitDate = date.split("-").map(_.trim)
    val splitTime = time.split(":").map(_.trim)

    val year = splitDate(2).toInt + 2000
    val month = splitDate(1).toInt
    val day = splitDate(0).toInt

    val hour = splitTime(0).toInt
    val minute = splitTime(1).toInt

    val zonedDate = ZonedDateTime.of(year, month, day, hour, minute, 0, 0, zoneId)

    zonedDate.toEpochSecond
  }
}

class MarineTrafficStream(
    filename: String,
    shipIds: Set[String],
    start: Long,
    end: Long
) extends EventStreamI {

  override def generateStream(): EventStream = {
    val eventStream = new EventStream()
    var totalCounter = 1
    val bufferedSource = io.Source.fromFile(filename)
    for (line <- bufferedSource.getLines) {
      val ev = line2Event(line, totalCounter)
      if (satisfyConstraints(ev)) {
        totalCounter += 1
        eventStream.addEvent(ev)
      }
    }

    eventStream
  }

  private def satisfyConstraints(ev: GenericEvent): Boolean = {
    if (ev.id == -1) false
    else {
      (ev.timestamp > start) &
        ((ev.timestamp < end) | (end == -1)) &
        ((shipIds.isEmpty) | shipIds.contains(ev.getValueOf("shipId").toString))
    }
  }

  private def line2Event(line: String, id: Int): GenericEvent = {
    val cols = line.split(",").map(_.trim)
    try {
      val shipId = s"${cols(0)}"
      val shipType = s"${cols(1)}"
      val speed = s"${cols(2)}".toDouble
      val lon = s"${cols(3)}".toDouble
      val lat = s"${cols(4)}".toDouble
      val course = s"${cols(5)}".toDouble
      val heading = s"${cols(6)}".toDouble
      val timestamp = date2Unix(s"${cols(7)}")
      val departurePort = s"${cols(8)}"
      val draught = s"${cols(9)}".toInt
      val arrivalTime = date2Unix(s"${cols(10)}")
      val arrivalPort = s"${cols(11)}"
      val ge = GenericEvent(id, "AIS", timestamp,
                            Map("shipId" -> shipId, "shipType" -> shipType, "speed" -> speed, "lon" -> lon, "lat" -> lat,
          "course" -> course, "heading" -> heading, "departurePort" -> departurePort, "draught" -> draught,
          "arrivalTime" -> arrivalTime, "arrivalPort" -> arrivalPort))
      ge
    } catch {
      case e: Exception => GenericEvent(-1, "NA", -1)
    }
  }

  private def date2Unix(str: String): Long = {
    val zoneId = ZoneId.of("UTC")

    val dateNtime = str.split(" ").map(_.trim)
    val date = dateNtime(0)
    val time = dateNtime(1)
    val splitDate = date.split("-").map(_.trim)
    val splitTime = time.split(":").map(_.trim)

    val year = splitDate(2).toInt + 2000
    val month = splitDate(1).toInt
    val day = splitDate(0).toInt

    val hour = splitTime(0).toInt
    val minute = splitTime(1).toInt

    val zonedDate = ZonedDateTime.of(year, month, day, hour, minute, 0, 0, zoneId)

    zonedDate.toEpochSecond
  }

}
