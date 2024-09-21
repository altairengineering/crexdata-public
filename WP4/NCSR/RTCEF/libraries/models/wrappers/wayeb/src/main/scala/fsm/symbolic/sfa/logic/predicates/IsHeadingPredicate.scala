package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class IsHeadingPredicate(arguments: List[String]) extends Predicate {
  val Norths = Set("A", "E", "I", "M", "Q", "U")
  val Easts = Set("B", "F", "J", "N", "R", "V")
  val Souths = Set("C", "G", "K", "O", "S", "W")
  val Wests = Set("D", "H", "L", "P", "T", "X")
  require(arguments.size == 1)
  val direction = arguments(0)
  override def evaluate(event: GenericEvent): Boolean = {
    val heading = event.getValueOf("heading").toString.toDouble
    val ev = direction match {
      case "North" => (heading >= 315.0) | (heading < 45.0)
      case "East" => (heading >= 45.0) & (heading < 135.0)
      case "South" => (heading >= 135.0) & (heading < 225.0)
      case "West" => (heading >= 225.0) & (heading < 315.0)
      case _ => throw new IllegalArgumentException
    }
    /*require(consistent(ev,direction,ge),"Event type: " + ge.eventType +
      " direction: " + direction +
      " eval: " + ev +
      " heading: " + ge.getValueOf("heading").toString)*/
    ev
  }

  private def consistent(ev: Boolean, dir: String, ge: GenericEvent): Boolean = {
    dir match {
      case "North" => if (Norths.contains(ge.eventType)) (ev == true) else (ev == false)
      case "East" => if (Easts.contains(ge.eventType)) (ev == true) else (ev == false)
      case "South" => if (Souths.contains(ge.eventType)) (ev == true) else (ev == false)
      case "West" => if (Wests.contains(ge.eventType)) (ev == true) else (ev == false)
    }
  }

  override def toString: String = "IsHeadingPredicate(" + list2Str(arguments, ",") + ")"

  private def isBetween(heading: Double, left: Double, right: Double): Boolean = (heading >= left) & (heading < right)

}
