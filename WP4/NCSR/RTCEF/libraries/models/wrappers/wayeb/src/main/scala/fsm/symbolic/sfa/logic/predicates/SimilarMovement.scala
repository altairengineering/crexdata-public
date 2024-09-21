package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str
import utils.SpatialUtils.distanceBetween

import scala.math.abs

case class SimilarMovement(arguments: List[String]) extends Predicate {
  private val maxDistance = arguments(0).toDouble
  private val maxSpeedDiff = arguments(1).toDouble
  private val maxHeadingDiff = arguments(2).toDouble

  override def evaluate(event: GenericEvent): Boolean = {
    val lon1 = event.getValueOf("lon1").toString.toDouble
    val lat1 = event.getValueOf("lat1").toString.toDouble
    val lon2 = event.getValueOf("lon2").toString.toDouble
    val lat2 = event.getValueOf("lat2").toString.toDouble
    val close = distanceBetween(lon1, lat1, lon2, lat2, 0, maxDistance)
    val speed1 = event.getValueOf("speed1").toString.toDouble
    val speed2 = event.getValueOf("speed2").toString.toDouble
    val similarSpeed = abs(speed1 - speed2) <= maxSpeedDiff
    val heading1 = event.getValueOf("heading1").toString.toDouble
    val heading2 = event.getValueOf("heading2").toString.toDouble
    val similarHeading = abs(heading1 - heading2) <= maxHeadingDiff
    (speed1 > 1.2 & speed1 < 10) & (speed2 > 1.2 & speed2 < 10) & close & similarSpeed & similarHeading
  }

  override def toString: String = "SimilarMovement(" + list2Str(arguments, ",") + ")"

}
