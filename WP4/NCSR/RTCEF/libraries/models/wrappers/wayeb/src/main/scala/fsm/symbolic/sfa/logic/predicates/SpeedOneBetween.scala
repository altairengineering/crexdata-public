package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class SpeedOneBetween(arguments: List[String]) extends Predicate {
  private val min_speed = arguments(0).toDouble
  private val max_speed = arguments(1).toDouble

  override def evaluate(event: GenericEvent): Boolean = {
    val speed = event.getValueOf("speed1").toString.toDouble
    (speed >= min_speed) & (speed < max_speed)
  }

  override def toString: String = "SpeedOneBetween(" + list2Str(arguments, ",") + ")"

}
