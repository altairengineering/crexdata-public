package scripts.sessionsNtests.misc

import java.io.FileWriter

import org.locationtech.jts.geom.{Coordinate, GeometryFactory, Point}

import scala.xml.XML

object kmlio {

  val patternsPath = "/home/zmithereen/src/Wayeb/patterns/datacron/maritime/approachingAllPorts"
  val preproc = false
  def main(args: Array[String]): Unit = {
    val ports = readPorts(args(0))
    val portsIds = ports.zipWithIndex
    val m = 0
    portsIds.foreach(pi => writePattern(pi._1.getX, pi._1.getY, m, pi._2, preproc))

  }

  def readPorts(fn: String): List[Point] = {
    val loadnode = XML.loadFile(fn)
    val portsStr = (loadnode \\ "coordinates").map(_.text).toList
    val ports = portsStr.map(p => p.split(",")).map(p => (p(0).toDouble, p(1).toDouble))
    println(ports)
    val geomfact = new GeometryFactory()
    val portsPoints = ports.map(p => geomfact.createPoint(new Coordinate(p._1, p._2)))
    println(portsPoints)
    println(portsPoints.size)
    portsPoints
  }

  def writePattern(
      lon: Double,
      lat: Double,
      m: Int,
      id: Int,
      preproc: Boolean
  ): String = {
    val pattern = if (preproc)
      ";(DSevenTenPredicate(" + id.toDouble + ")," +
        "DFiveSevenPredicate(" + id.toDouble + ")," +
        "*(DFiveSevenPredicate(" + id.toDouble + "))," +
        "InPortPredicate(" + id.toDouble + "))" +
        "{order:" + m + "}{partitionBy:mmsi}"
    else
      ";(DistanceBetweenPredicate(" + lon + "," + lat + ",7.0,10.0)," +
        "DistanceBetweenPredicate(" + lon + "," + lat + ",5.0,7.0)," +
        "*(DistanceBetweenPredicate(" + lon + "," + lat + ",5.0,7.0))," +
        "WithinCirclePredicate(" + lon + "," + lat + ",5.0))" +
        "{order:" + m + "}{partitionBy:mmsi}"

    val fn = patternsPath + "/" + "approachingAllPorts_m_" + m + "_" + preproc + ".cepl"
    val writer = new FileWriter(fn, true)
    writer.write(pattern + "\n&\n")
    writer.close()
    pattern
  }
}
