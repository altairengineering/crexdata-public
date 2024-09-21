package scripts.data.caviar

object CaviarGroup {
  def apply(id1: Int, id2: Int, frame: Int, situation: String, context: String): CaviarGroup =
    new CaviarGroup(id1, id2, frame, situation, context)
}

class CaviarGroup(
    val id1: Int,
    val id2: Int,
    val frame: Int,
    val situation: String,
    val context: String
) {

}
