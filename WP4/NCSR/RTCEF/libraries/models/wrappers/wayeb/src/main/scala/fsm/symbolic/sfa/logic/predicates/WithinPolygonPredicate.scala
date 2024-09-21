package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import org.locationtech.jts.geom.{Coordinate, GeometryFactory}
import stream.GenericEvent
import utils.StringUtils.list2Str

case class WithinPolygonPredicate(arguments: List[String]) extends Predicate {
  private val polygonToCheck = arguments.head
  private val geomfact = new GeometryFactory()
  private val areaArray: Array[(Double, Double)] = {
    polygonToCheck match {
      case "FishingArea" => {
        val area_fn = System.getenv("WAYEB_HOME") + "/scripts/data/maritime/static/" + "brest_fishing_areas_110_simplified.txt"
        val source = scala.io.Source.fromFile(area_fn)
        val lines = try source.mkString finally source.close()
        val coords = lines.split(" ").toList.map(f => f.split(",")).map(f => (f(0).toDouble, f(1).toDouble)).toArray
        coords
      }
      case "AnchorageArea" => {
        val area_fn = System.getenv("WAYEB_HOME") + "/scripts/data/maritime/static/" + "anchorageArea1.csv"
        val source = scala.io.Source.fromFile(area_fn)
        val lines = try source.mkString finally source.close()
        val coords = lines.split(",").toList.map(f => f.split(" ")).map(f => (f(0).toDouble, f(1).toDouble)).toArray
        coords
      }
    }

  }
  private val areaPoly = geomfact.createPolygon(areaArray.map(c => new Coordinate(c._1, c._2)))

  override def evaluate(event: GenericEvent): Boolean = {
    val lon = event.getValueOf("lon").toString.toDouble
    val lat = event.getValueOf("lat").toString.toDouble
    val point = geomfact.createPoint(new Coordinate(lon, lat))
    point.intersects(areaPoly)
  }

  override def toString: String = "WithinPolygonPredicate(" + list2Str(arguments, ",") + ")"

}
