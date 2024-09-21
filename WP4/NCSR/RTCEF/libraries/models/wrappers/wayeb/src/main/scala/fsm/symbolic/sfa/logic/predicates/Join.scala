package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class Join(args: List[String]) extends Predicate {

  override def evaluate(event: GenericEvent): Boolean = {

    val situation = event.getValueOf("situation").toString
    situation.equals("joining")

  }

  override def toString: String = "Join(" + list2Str(args, ",") + ")"

}
