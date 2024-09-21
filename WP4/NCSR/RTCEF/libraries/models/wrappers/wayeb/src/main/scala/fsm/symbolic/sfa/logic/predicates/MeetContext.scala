package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class MeetContext(args: List[String]) extends Predicate {

  override def evaluate(event: GenericEvent): Boolean = {

    val cxt = event.getValueOf("cxt").toString

    if (cxt.equals("meeting"))
      return true
    return false

  }

  override def toString: String = "MeetContext(" + list2Str(args, ",") + ")"

}
