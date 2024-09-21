package stream.domain.archived

import stream.GenericEvent

class FeedzaiEvent(
    id: Int,
    timestamp: Int,
    eventType: String,
    pan: String,
    isFraud: Boolean
) extends GenericEvent(id,
                       eventType.toString,
                       timestamp.toLong,
                       Map[String, Any]("pan" -> pan, "isFraud" -> isFraud)
) {

  def getId: Int = id
  def getTimestamp: Int = timestamp
  def getType: String = eventType

  def getAttribute(which: String): String = {
    if (which.equalsIgnoreCase("id")) id.toString
    else if (which.equalsIgnoreCase("timestamp")) timestamp.toString
    else if (which.equalsIgnoreCase("type")) eventType.toString
    else if (which.equalsIgnoreCase("pan")) pan
    else if (which.equalsIgnoreCase("isFraud")) isFraud.toString
    //else if (which.equalsIgnoreCase("fraudType")) fraudType.toString
    else throw new IllegalArgumentException("FeedzaiEvent does not have attribute: " + which)
  }

  override def hasAttribute(attribute: String): Boolean = {
    if (attribute.equalsIgnoreCase("id")) return true
    if (attribute.equalsIgnoreCase("timestamp")) return true
    if (attribute.equalsIgnoreCase("type")) return true
    if (attribute.equalsIgnoreCase("pan")) return true
    if (attribute.equalsIgnoreCase("isFraud")) return true
    false
  }

  def getValues: List[Any] = List(eventType, timestamp, pan, isFraud)

  override def toString: String = {
    id + "\t|\t" + timestamp + "\t|\t" + eventType + "\t|\t" + pan + "\t|\t" + isFraud //+ "\t|\t" + fraudType
  }
}
