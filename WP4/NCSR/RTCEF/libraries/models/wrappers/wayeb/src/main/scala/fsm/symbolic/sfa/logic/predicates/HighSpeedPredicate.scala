package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class HighSpeedPredicate(arguments: List[String]) extends Predicate {
  override def evaluate(event: GenericEvent): Boolean = {
    val speed = event.getValueOf("speed").toString.toDouble
    speed >= 20.0
  }

  override def toString: String = "HighSpeedPredicate(" + list2Str(arguments, ",") + ")"

}
