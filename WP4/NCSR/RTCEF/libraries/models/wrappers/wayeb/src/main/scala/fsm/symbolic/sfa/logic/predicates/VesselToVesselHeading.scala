package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class VesselToVesselHeading(arguments: List[String]) extends Predicate {
  private val min = arguments(0).toDouble
  private val max = arguments(1).toDouble

  override def evaluate(event: GenericEvent): Boolean = {
    val heading1 = event.getValueOf("heading1").toString.toDouble
    val heading2 = event.getValueOf("heading2").toString.toDouble
    val headingDiff = scala.math.abs(heading1 - heading2)
    (headingDiff >= min) & (headingDiff < max)
  }

  override def toString: String = "VesselToVesselHeading(" + list2Str(arguments, ",") + ")"

}
