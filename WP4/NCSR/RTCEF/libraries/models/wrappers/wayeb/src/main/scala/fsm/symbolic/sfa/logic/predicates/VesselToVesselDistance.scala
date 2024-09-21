package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str
import utils.SpatialUtils.distanceBetween

case class VesselToVesselDistance(arguments: List[String]) extends Predicate {
  private val min = arguments(0).toDouble
  private val max = arguments(1).toDouble

  override def evaluate(event: GenericEvent): Boolean = {
    val lon1 = event.getValueOf("lon1").toString.toDouble
    val lat1 = event.getValueOf("lat1").toString.toDouble
    val lon2 = event.getValueOf("lon2").toString.toDouble
    val lat2 = event.getValueOf("lat2").toString.toDouble
    distanceBetween(lon1, lat1, lon2, lat2, min, max)
  }

  override def toString: String = "VesselToVesselDistance(" + list2Str(arguments, ",") + ")"
}
