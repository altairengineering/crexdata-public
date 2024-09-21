package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class Context(args: List[String]) extends Predicate {
  private val which = args(0)
  private val contextToCheck = args(1)

  override def evaluate(event: GenericEvent): Boolean = {
    val context =
      which match {
        case "one" => event.getValueOf("con1").toString
        case "two" => event.getValueOf("con2").toString
        case "group" => event.getValueOf("gcon").toString
        case _ => throw new IllegalArgumentException
      }
    context.equalsIgnoreCase(contextToCheck)
  }

  override def toString: String = "Context(" + list2Str(args, ",") + ")"

}
