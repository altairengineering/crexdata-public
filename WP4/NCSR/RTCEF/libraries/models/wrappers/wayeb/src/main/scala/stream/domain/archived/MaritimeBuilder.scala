/*package stream.domain.maritime.datacron.maritime

import builder.Builder
import data.InstantaneousEvent
import stream.GenericEvent

import scala.collection.mutable

class MaritimeBuilder extends Builder[GenericEvent] {
  private val collection = mutable.ArrayBuffer.empty[GenericEvent]
  private var idCounter = 0

  override def +=(event: InstantaneousEvent): Unit = {
    try {
      val mmsi = event.args(0)
      val speed = event.args(1).toDouble
      val heading = event.args(2).toDouble
      val lon = event.args(3).toDouble
      val lat = event.args(4).toDouble
      val timestamp = event.time
      idCounter += 1
      val ge = GenericEvent(idCounter, "SampledCritical", timestamp,
        Map("mmsi"->mmsi,"speed"->speed,"lon"->lon,"lat"->lat, "heading"->heading))
      collection += ge
    }
    catch {
      case e: Exception => throw new IllegalArgumentException
    }
  }

  override def result: GenericEvent = collection.head

  override def clear(): Unit = collection.clear()

}*/
