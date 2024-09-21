package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class Split(args: List[String]) extends Predicate {

  override def evaluate(event: GenericEvent): Boolean = {

    val situation = event.getValueOf("situation").toString
    situation.equals("split up")

  }

  override def toString: String = "Split(" + list2Str(args, ",") + ")"

}
