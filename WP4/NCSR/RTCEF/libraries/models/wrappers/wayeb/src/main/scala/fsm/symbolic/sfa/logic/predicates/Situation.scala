package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class Situation(args: List[String]) extends Predicate {
  private val which = args(0)
  private val situationToCheck = args(1)

  override def evaluate(event: GenericEvent): Boolean = {
    val situation =
      which match {
        case "one" => event.getValueOf("sit1").toString
        case "two" => event.getValueOf("sit2").toString
        case "group" => event.getValueOf("gsit").toString
        case _ => throw new IllegalArgumentException
      }
    situation.equalsIgnoreCase(situationToCheck)
  }

  override def toString: String = "Situation(" + list2Str(args, ",") + ")"

}
