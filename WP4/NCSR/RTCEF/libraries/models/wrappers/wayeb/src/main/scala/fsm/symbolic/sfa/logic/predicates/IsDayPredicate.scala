package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class IsDayPredicate(arguments: List[String]) extends Predicate {
  override def evaluate(event: GenericEvent): Boolean = {
    val day = event.getValueOf("day").toString.toDouble.toInt
    val dayCheck = arguments(0).toDouble.toInt
    day == dayCheck
  }

  override def toString: String = "IsDay(" + list2Str(arguments, ",") + ")"

}
