package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class VesselToVesselSpeed(arguments: List[String]) extends Predicate {
  private val min = arguments(0).toDouble
  private val max = arguments(1).toDouble

  override def evaluate(event: GenericEvent): Boolean = {
    val speed1 = event.getValueOf("speed1").toString.toDouble
    val speed2 = event.getValueOf("speed2").toString.toDouble
    val speedDiff = scala.math.abs(speed1 - speed2)
    (speedDiff >= min) & (speedDiff < max)
  }

  override def toString: String = "VesselToVesselSpeed(" + list2Str(arguments, ",") + ")"

}
