package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import org.locationtech.jts.geom.{Coordinate, GeometryFactory}
import org.locationtech.jts.operation.distance.DistanceOp
import stream.GenericEvent
import utils.StringUtils.list2Str

case class DistToPolyPredicate(arguments: List[String]) extends Predicate {
  val innerRadius = arguments(0).toDouble
  val outerRadius = arguments(1).toDouble
  private val geomfact = new GeometryFactory()
  private val FishingAreaArray: Array[(Double, Double)] = {
    val area_fn = "/home/zmithereen/data/maritime/cyril/brest_fishing_areas_110_simplified.txt"
    val source = scala.io.Source.fromFile(area_fn)
    val lines = try source.mkString finally source.close()
    val coords = lines.split(" ").toList.map(f => f.split(",")).map(f => (f(0).toDouble, f(1).toDouble)).toArray
    coords
  }
  private val FishingAreaPoly = geomfact.createPolygon(FishingAreaArray.map(c => new Coordinate(c._1, c._2)))

  override def evaluate(event: GenericEvent): Boolean = {
    val lon = event.getValueOf("lon").toString.toDouble
    val lat = event.getValueOf("lat").toString.toDouble
    val point = geomfact.createPoint(new Coordinate(lon, lat))
    val distance = DistanceOp.distance(point, FishingAreaPoly.getExteriorRing)
    (distance > innerRadius) & (distance < outerRadius)
  }

  override def toString: String = "DistToPolyPredicate(" + list2Str(arguments, ",") + ")"

}
