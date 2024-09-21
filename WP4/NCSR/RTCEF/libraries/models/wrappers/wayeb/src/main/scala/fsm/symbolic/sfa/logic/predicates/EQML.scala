package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class EQML(arguments: List[String]) extends Predicate {
  override def evaluate(event: GenericEvent): Boolean = {
    require(arguments.size == 2)
    val field = arguments.head match {
      case "EntryNearcoast" => "entryNearcoast"
      case "EntryNearcoast5k" => "entryNearcoast5k"
      case "EntryFishing" => "entryFishing"
      case "EntryNatura" => "entryNatura"
      case "EntryNearports" => "entryNearports"
      case "EntryAnchorage" => "entryAnchorage"
      case "ExitNearcoast" => "exitNearcoast"
      case "ExitNearcoast5k" => "exitNearcoast5k"
      case "ExitFishing" => "exitFishing"
      case "ExitNatura" => "exitNatura"
      case "ExitNearports" => "exitNearports"
      case "ExitAnchorage" => "exitAnchorage"
      case "StopStart" => "stop_start"
      case "StopEnd" => "stop_end"
      case "SlowMotionStart" => "slow_motion_start"
      case "SlowMotionEnd" => "slow_motion_end"
      case "GapEnd" => "gap_end"
      case "GapStart" => "gap_start"
      case "ChangeInHeading" => "change_in_heading"
      case "ChangeInSpeedStart" => "change_in_speed_start"
      case "ChangeInSpeedEnd" => "change_in_speed_end"
    }
    val variableValue = event.getValueOf(field).toString.toDouble
    val constant = arguments(1).toDouble
    variableValue == constant
  }

  override def toString: String = "EQML(" + list2Str(arguments, ",") + ")"

}
