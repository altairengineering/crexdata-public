package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class IsHourPredicate(arguments: List[String]) extends Predicate {
  override def evaluate(event: GenericEvent): Boolean = {
    val hour = event.getValueOf("hour").toString.toDouble.toInt
    val hourCheck = arguments(0).toInt
    hour == hourCheck
  }

  override def toString: String = "IsHour(" + list2Str(arguments, ",") + ")"

}
