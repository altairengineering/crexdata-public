package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class DSevenTenPredicate(arguments: List[String]) extends Predicate {
  val port = arguments(0).toDouble.toInt

  override def evaluate(event: GenericEvent): Boolean = {
    val d710PortsStr = event.getValueOfMap("d710").toString
    if (d710PortsStr == "") false
    else {
      val d710Ports = d710PortsStr.split(",").map(_.trim.toInt)
      d710Ports.contains(port)
    }

  }

  override def toString: String = "DSevenTenPredicate(" + list2Str(arguments, ",") + ")"

}
