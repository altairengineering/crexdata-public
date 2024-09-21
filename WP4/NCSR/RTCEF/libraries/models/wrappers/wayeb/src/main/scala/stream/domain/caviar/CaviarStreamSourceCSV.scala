package stream.domain.caviar

import stream.{GenericEvent, ResetEvent}
import stream.source.CSVStreamSource

object CaviarStreamSourceCSV {
  /**
    * Construvtor for caviar stream source.
    *
    * @param filename The path to the file.
    * @return The stream source.
    */
  def apply(filename: String): CaviarStreamSourceCSV = new CaviarStreamSourceCSV(filename)
}

/**
  * Stream source for caviar videos given in a single csv file.
  *
  * @param filename The path to the file.
  */
class CaviarStreamSourceCSV(filename: String) extends CSVStreamSource(filename) {

  override def line2Event(
                           line: Seq[String],
                           id: Int
                         ): GenericEvent = {
    val frameNo = line(0).toInt
    if (frameNo == -1) {
      val id1 = line(1).toInt
      val id2 = line(2).toInt
      val idpair = id1 + "-" + id2
      ResetEvent(Map("idpair" -> idpair))
    } else {
      val object1Str = line(3).split(";")
      val object2Str = line(4).split(";")
      val groupStr = line(5).split(";")
      val id1 = object1Str(0).toInt
      val id2 = object2Str(0).toInt
      val idpair = id1 + "-" + id2
      val x1 = object1Str(1).toInt
      val y1 = object1Str(2).toInt
      val x2 = object2Str(1).toInt
      val y2 = object2Str(2).toInt
      val sit1 = object1Str(7)
      val con1 = object1Str(8)
      val sit2 = object2Str(7)
      val con2 = object2Str(8)
      val dist = groupStr(0).toInt
      val (gsit, gcon) = if (groupStr.length > 1) (groupStr(1), groupStr(2)) else ("na", "na")
      val nextCETimestamp = if (line.size > 6) line(6).toInt else -1
      val ge = GenericEvent(id, "Caviar", frameNo,
                            Map("idpair" -> idpair, "x1" -> x1, "y1" -> y1, "sit1" -> sit1, "con1" -> con1,
          "x2" -> x2, "y2" -> y2, "sit2" -> sit2, "con2" -> con2, "dist" -> dist, "gsit" -> gsit, "gcon" -> gcon,
          "nextCETimestamp" -> nextCETimestamp))
      ge
    }
  }

}
