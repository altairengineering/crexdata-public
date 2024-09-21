package scripts.data.maritime

object CriticalEvent {
  def apply(
             id: Int,
             timestamp: Long,
             lon: Double,
             lat: Double,
             annotation: String,
             speed: Double,
             heading: Double,
             turn: Double,
             course: Double,
             valid: Boolean
           ): CriticalEvent = new CriticalEvent(id, timestamp, lon, lat, annotation, speed, heading, turn, course, valid)

  def apply(id: Int): CriticalEvent = new CriticalEvent(id, -1, -1.0, -1.0, "-1", -1, -1, -1, -1, true)
}

class CriticalEvent(
                     val id: Int,
                     val timestamp: Long,
                     val lon: Double,
                     val lat: Double,
                     val annotation: String,
                     val speed: Double,
                     val heading: Double,
                     val turn: Double,
                     val course: Double,
                     val valid: Boolean
                   ) extends Serializable {

  override def toString: String = "id:" + id.toString + " " + bitmap2Str(annotation) + " t:" + timestamp + " lon:" + lon + " lat:" + lat + " sp:" + speed.toInt + " h:" + heading.toInt

  def toRow: List[String] = List(timestamp.toString, id.toString, lon.toString, lat.toString, speed.toString, heading.toString, course.toString, annotation)

  private def bitmap2Str(bitmap: String): String = {
    bitmap match {
      case "00000001" => "stop_start"
      case "00000010" => "stop_end"
      case "00000100" => "slow_motion_start"
      case "00001000" => "slow_motion_end"
      case "00010000" => "gap_end"
      case "00100000" => "change_in_heading"
      case "01000000" => "change_in_speed_start"
      case "10000000" => "change_in_speed_end"
      case _ => "unknown"
    }
  }

}
