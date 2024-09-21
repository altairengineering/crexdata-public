package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import org.locationtech.jts.geom.{Coordinate, GeometryFactory}
import stream.GenericEvent
import utils.SpatialUtils.projectPoint

class HeadingVesselsToMeet(arguments: List[String]) extends Predicate {
  private val geomfact = new GeometryFactory()
  private val projectionSeconds = arguments(0).toDouble

  override def evaluate(event: GenericEvent): Boolean = {

    val lon1 = event.getValueOfMap("lon1").toString.toDouble
    val lat1 = event.getValueOfMap("lat1").toString.toDouble
    val speed1 = event.getValueOfMap("speed1").toString.toDouble
    val heading1 = event.getValueOfMap("heading1").toString.toDouble
    val lon2 = event.getValueOfMap("lon2").toString.toDouble
    val lat2 = event.getValueOfMap("lat2").toString.toDouble
    val speed2 = event.getValueOfMap("speed2").toString.toDouble
    val heading2 = event.getValueOfMap("heading2").toString.toDouble
    val (new_lon1, new_lat1) = projectPoint(lon1, lat1, projectionSeconds.toInt, speed1, heading1)
    val (new_lon2, new_lat2) = projectPoint(lon2, lat2, projectionSeconds.toInt, speed2, heading2)
    val lineCoords1 = Array[Coordinate](new Coordinate(lon1, lat1), new Coordinate(new_lon1, new_lat1))
    val lineCoords2 = Array[Coordinate](new Coordinate(lon2, lat2), new Coordinate(new_lon2, new_lat2))
    val line1 = geomfact.createLineString(lineCoords1)
    val line2 = geomfact.createLineString(lineCoords2)
    line1.intersects(line2)
  }

  override def toString: String = "HeadingVesselsToMeet(" + projectionSeconds.toString + ")"

}
