package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class InPortPredicate(arguments: List[String]) extends Predicate {
  val port = arguments(0).toDouble.toInt

  override def evaluate(event: GenericEvent): Boolean = {
    val inPortsStr = event.getValueOfMap("within").toString
    if (inPortsStr == "") false
    else {
      val inPorts = inPortsStr.split(",").map(_.trim.toInt)
      inPorts.contains(port)
    }
  }

  override def toString: String = "InPortPredicate(" + list2Str(arguments, ",") + ")"

}
