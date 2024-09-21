package stream.source

import com.typesafe.scalalogging.LazyLogging
import stream.{GenericEvent, ResetEvent}

object GenericCSVStreamSource {
  def apply(fn: String): GenericCSVStreamSource = new GenericCSVStreamSource(fn)
}

/**
  * A generic (not domain-dependent) source for CSV files. First column is the event type. Second column the timestamp.
  *
  * @param filename The path to the file.
  */
class GenericCSVStreamSource private[stream](filename: String) extends CSVStreamSource(filename) with LazyLogging {

  /**
    * First column is the event type. Second column the timestamp.
    *
    * @param line A line, as a sequence of strings.
    * @param id The new event's unique id.
    * @return The line converted to an event.
    */
  override def line2Event(
                           line: Seq[String],
                           id: Int
                         ): GenericEvent = {
    try {
      val eventType = line.head
      val timestamp = line(1).toLong
      if (timestamp == -1) ResetEvent()
      else {
        val ge = GenericEvent(id, eventType, timestamp)
        ge
      }
    } catch {
      case _: Exception => {
        logger.warn("COULD NOT PARSE LINE " + line)
        throw new Error
      }
    }
  }

  /*override def emitEvents(mode: EmitMode): EventStream = {
    val eventStream = new EventStream()
    var totalCounter = 0
    var eventTypes = Set.empty[String]
    val reader = CSVReader.open(filename)
    val it = reader.iterator
    while (it.hasNext) {
      val line = it.next()
      totalCounter += 1
      val newEvent = line2Event(line, totalCounter)
      mode match {
        case EmitMode.BUFFER => {
          eventStream.addEvent(newEvent)
          eventTypes += newEvent.eventType
        }
        case EmitMode.ONLINE => send2Listeners(newEvent)
      }
    }
    reader.close()
    eventStream.setEventTypes(eventTypes)
    eventStream
  }*/

}
