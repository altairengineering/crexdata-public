package stream.domain.archived

import com.typesafe.scalalogging.LazyLogging
import stream.GenericEvent
import stream.array.{EventStream, EventStreamI}

object VodafoneStream {
  def apply(filename: String): VodafoneStream = new VodafoneStream(filename)
}

class VodafoneStream(filename: String) extends EventStreamI with LazyLogging {
  override def generateStream(): EventStream = {
    val eventStream = new EventStream()
    val bufferedSource = io.Source.fromFile(filename)
    var totalCounter = 1
    for (line <- bufferedSource.getLines) {
      val cols = line.split(";").map(_.trim)
      val company = s"${cols(0)}".toString
      val engineStatus = s"${cols(3)}".toInt
      val lon = s"${cols(6)}".toDouble
      val lat = s"${cols(7)}".toDouble
      val vehicle = s"${cols(1)}".toString
      val id = totalCounter
      totalCounter += 1
      val ge = GenericEvent(id, "Vodafone", id,
                            Map("vehicle" -> vehicle, "company" -> company, "engineStatus" -> engineStatus, "lon" -> lon, "lat" -> lat))
      eventStream.addEvent(ge)
    }
    eventStream
  }
}
