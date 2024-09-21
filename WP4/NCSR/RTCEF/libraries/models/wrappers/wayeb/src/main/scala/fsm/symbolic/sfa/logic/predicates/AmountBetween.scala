package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class AmountBetween(arguments: List[String]) extends Predicate {
  private val min = arguments(0).toDouble
  private val max = arguments(1).toDouble

  override def evaluate(event: GenericEvent): Boolean = {
    val amount = event.getValueOf("amount").toString.toDouble
    amount >= min & amount < max
  }

  override def toString: String = "AmountBetween(" + list2Str(arguments, ",") + ")"
}
