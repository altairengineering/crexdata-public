package stream.domain.vodafone

import stream.source.{EmitMode, CSVStreamSource}
import stream.source.EmitMode.EmitMode
import stream.{GenericEvent, ResetEvent}
import stream.array.EventStream

object VodafoneStreamSourceCSV {
  /**
    * Constructor for vodafone stream sources.
    *
    * @param filename The path to the file.
    * @return The stream source.
    */
  def apply(filename: String): VodafoneStreamSourceCSV = new VodafoneStreamSourceCSV(filename)
}

/**
  * Stream source for vodafone files.
  *
  * @param filename The path to the file.
  */
class VodafoneStreamSourceCSV(filename: String) extends CSVStreamSource(filename) {

  //TODO: use csv reader (with ; as separator) and do not override emitEvents
  override def emitEvents(mode: EmitMode): EventStream = {
    val bufferedSource = io.Source.fromFile(filename)
    var totalCounter = 1
    val eventStream = new EventStream()
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
      mode match {
        case EmitMode.BUFFER => eventStream.addEvent(ge)
        case EmitMode.ONLINE => send2Listeners(ge)
      }
    }
    eventStream
  }

  override protected def line2Event(
                                     line: Seq[String],
                                     id: Int
                                   ): GenericEvent = ResetEvent()
}
