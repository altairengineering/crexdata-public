package scripts.data.caviar

object CaviarObject {
  def apply(id: Int, frame: Int, x: Int, y: Int, orx: Int, ory: Int, oran: Int, gaze: Int, situation: String, context: String): CaviarObject =
    new CaviarObject(id, frame, x, y, orx, ory, oran, gaze, situation, context)
}

class CaviarObject(
    val id: Int,
    val frame: Int,
    val x: Int,
    val y: Int,
    val orientationx: Int,
    val orientationy: Int,
    val orienationAngle: Int,
    val gaze: Int,
    val situation: String,
    val context: String
) {

  override def toString: String =
    id.toString + ";" +
      x.toString + ";" +
      y.toString + ";" +
      orientationx + ";" +
      orientationy + ";" +
      orienationAngle + ";" +
      gaze.toString + ";" +
      situation + ";" +
      context

}
