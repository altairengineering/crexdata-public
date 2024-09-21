package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class DFiveSevenPredicate(arguments: List[String]) extends Predicate {
  val port = arguments(0).toDouble.toInt

  override def evaluate(event: GenericEvent): Boolean = {
    val d57PortsStr = event.getValueOfMap("d57").toString
    if (d57PortsStr == "") false
    else {
      val d57Ports = d57PortsStr.split(",").map(_.trim.toInt)
      d57Ports.contains(port)
    }

  }

  override def toString: String = "D57Predicate(" + list2Str(arguments, ",") + ")"
}
