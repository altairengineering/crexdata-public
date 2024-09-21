package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class IsTimePredicate(arguments: List[String]) extends Predicate {
  override def evaluate(event: GenericEvent): Boolean = {
    val hour = event.getValueOf("hour").toString.toDouble.toInt
    val timeOfDay = arguments(0)
    timeOfDay match {
      case "Morning" => (hour >= 6) & (hour < 12)
      case "Noon" => (hour >= 12) & (hour < 18)
      case "Evening" => (hour >= 18) & (hour < 24)
      case "Night" => (hour >= 24) & (hour < 6)
    }

    /*
    timeOfDay match {
      case "EarlyMorning" => (hour >= 6) & (hour < 8)
      case "MidMorning" => (hour >= 8) & (hour < 10)
      case "LateMorning" => (hour >= 10) & (hour < 12)
      case "EarlyNoon" => (hour >= 12) & (hour < 14)
      case "MidNoon" => (hour >= 14) & (hour < 16)
      case "LateNoon" => (hour >= 16) & (hour < 18)
      case "EarlyEvening" => (hour >= 18) & (hour < 20)
      case "MidEvening" => (hour >= 20) & (hour < 22)
      case "LateEvening" => (hour >= 22) & (hour < 24)
      case "EarlyNight" => (hour >= 24) & (hour < 2)
      case "MidNight" => (hour >= 2) & (hour < 4)
      case "LateNight" => (hour >= 4) & (hour < 6)
    }
     */
  }

  override def toString: String = "IsTime(" + list2Str(arguments, ",") + ")"
}
