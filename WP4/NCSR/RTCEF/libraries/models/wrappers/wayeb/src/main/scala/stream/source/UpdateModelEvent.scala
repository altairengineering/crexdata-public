package stream.source

import stream.GenericEvent

/**
  * Special event. This event indicates that there are no more events in the stream.
  * Useful so as to know when to shutdown the forecasting engine.
  */
final class UpdateModelEvent(id:Int, timestamp:Long, modelInfo:Map[String, String])
  extends GenericEvent(id, "UpdateModel", timestamp, modelInfo)
