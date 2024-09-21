package stream.source

import play.api.libs.json.{JsObject, Json}
import stream.GenericEvent
import stream.array.EventStream
import stream.source.EmitMode.EmitMode

object JsonStreamSource {
  def apply(fn: String): JsonStreamSource = new JsonStreamSource(fn)
}

/**
  * Stream source for JSON files. Every event attribute in the JSON event is mapped to an attribute of the generic
  * event. In BUFFER mode, events are stored in an array of events. In ONLINE mode, events are sent to listeners.
  *
  * @param filename The path to the file.
  */
class JsonStreamSource(filename: String) extends StreamSource {

  override protected def emitEvents(mode: EmitMode): EventStream = {
    val bufferedSource = io.Source.fromFile(filename)
    var totalCounter = 0L
    val eventStream = new EventStream()
    for (line <- bufferedSource.getLines) {
      val map = Json.parse(line).as[JsObject].value.toMap
      val timestamp = map.getOrElse("timestamp", totalCounter).toString.toLong

      totalCounter += 1
      val ge = GenericEvent(totalCounter.toInt, "GenericJson", timestamp, map)
      mode match {
        case EmitMode.BUFFER => eventStream.addEvent(ge)
        case EmitMode.ONLINE => send2Listeners(ge)
      }
    }
    eventStream
  }

}
