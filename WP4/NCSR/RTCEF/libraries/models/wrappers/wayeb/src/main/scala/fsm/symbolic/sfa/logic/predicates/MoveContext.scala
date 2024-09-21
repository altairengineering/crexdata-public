package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class MoveContext(args: List[String]) extends Predicate {

  override def evaluate(event: GenericEvent): Boolean = {

    val cxt1 = event.getValueOf("cxt1").toString
    val cxt2 = event.getValueOf("cxt2").toString

    if (cxt1.equals("walking") && cxt2.equals("walking"))
      return true
    return false

  }

  override def toString: String = "MoveContext(" + list2Str(args, ",") + ")"

}
