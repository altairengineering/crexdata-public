package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class IsConfigPredicate(arguments: List[String]) extends Predicate {
  override def evaluate(event: GenericEvent): Boolean = {
    val config = arguments(0).toDouble.toInt
    val configCheck = event.getValueOf("config").toString.toInt
    config == configCheck
  }

  override def toString: String = "IsConfig(" + list2Str(arguments, ",") + ")"

}
