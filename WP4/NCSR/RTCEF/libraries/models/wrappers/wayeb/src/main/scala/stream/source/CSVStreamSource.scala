package stream.source

import com.github.tototoshi.csv.CSVReader
import stream.GenericEvent
import stream.array.EventStream
import stream.source.EmitMode.EmitMode

/**
  * Stream source for CSV files.
  *
  * @param filename The path to the file.
  */
abstract class CSVStreamSource(filename: String) extends StreamSource {

  /**
    * After reading every line, it either sends it (as an event) to the listeners if in ONLINE mode or stores it to an
    * event stream if in BUFFER mode.
    *
    * @param mode The mode, BUFFER or ONLINE.
    * @return The stream as an array of events.
    */
  override def emitEvents(mode: EmitMode): EventStream = {
    val eventStream = new EventStream()
    var totalCounter = 1
    var eventTypes = Set.empty[String]
    val reader = CSVReader.open(filename)
    val it = reader.iterator
    while (it.hasNext) {
      val line = it.next()
      val newEvent = line2Event(line, totalCounter)
      totalCounter += 1
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
  }

  /**
    * Every concrete CSV stream source must implement this in order to determine how each line is to be converted to an
    * event.
    *
    * @param line A line, as a sequence of strings.
    * @param id The new event's unique id.
    * @return The line converted to an event.
    */
  protected def line2Event(
                            line: Seq[String],
                            id: Int
                          ): GenericEvent

}
