package scripts.data.maritime

object MaritimeUtils {
  def str2CriticalEvent(line: String): CriticalEvent = {
    val cols = line.split(",").map(_.trim)
    try {
      val id = s"${cols(0)}".toInt
      val timestamp = s"${cols(1)}".toLong / 1000
      val lon = s"${cols(2)}".toDouble
      val lat = s"${cols(3)}".toDouble
      val annotation = s"${cols(4)}".toString
      val speed = s"${cols(5)}".toDouble
      val heading = s"${cols(6)}".toDouble
      val turn = s"${cols(7)}".toDouble
      val course = s"${cols(8)}".toDouble
      new CriticalEvent(id, timestamp, lon, lat, annotation, speed, heading, turn, course, true)
    } catch {
      case e: Exception => new CriticalEvent(0, 0, 0, 0, "", 0, 0, 0.0, 0.0, false)
    }
  }
}
