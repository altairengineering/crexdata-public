package stream.domain.archived

import stream.GenericEvent
import stream.source.CSVStreamSource

object MaritimePairedStreamSourceCSV {
  def apply(filename: String): MaritimePairedStreamSourceCSV = new MaritimePairedStreamSourceCSV(filename)
}

class MaritimePairedStreamSourceCSV(filename: String) extends CSVStreamSource(filename) {


  override def line2Event(
                           line: Seq[String],
                           id: Int
                         ): GenericEvent = {
    try {
      val timestamp = line(0).toLong
      val eventType = line(1)
      val pairId = line(2)

      val lon1 = line(3).toDouble
      val lat1 = line(4).toDouble
      val speed1 = line(5).toDouble
      val heading1 = line(6).toDouble

      val lon2 = line(7).toDouble
      val lat2 = line(8).toDouble
      val speed2 = line(9).toDouble
      val heading2 = line(10).toDouble

      val ge = GenericEvent(id, eventType, timestamp,
                            Map("pairId" -> pairId,
          "lon1" -> lon1,
          "lat1" -> lat1,
          "speed1" -> speed1,
          "heading1" -> heading1,
          "lon2" -> lon2,
          "lat2" -> lat2,
          "speed2" -> speed2,
          "heading2" -> heading2
        ))
      ge
    } catch {
      case e: Exception => {
        throw new Error("Could not parse line " + line)
      }
    }
  }

}
