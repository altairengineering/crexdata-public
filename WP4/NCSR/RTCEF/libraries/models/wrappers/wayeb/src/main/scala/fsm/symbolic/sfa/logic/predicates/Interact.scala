package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class Interact(args: List[String]) extends Predicate {

  override def evaluate(event: GenericEvent): Boolean = {

    val situation = event.getValueOf("situation").toString
    situation.equals("interacting")

  }

  override def toString: String = "Interact(" + list2Str(args, ",") + ")"

}
