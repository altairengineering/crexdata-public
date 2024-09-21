package stream.domain.text

import stream.GenericEvent
import stream.source.CSVStreamSource

object TextStreamSourceCSV {
  /**
    * Constructor for text stream source.
    *
    * @param fn The path to the file.
    * @return The stream source.
    */
  def apply(fn: String): TextStreamSourceCSV = new TextStreamSourceCSV(fn)
}

/**
  * Stream source for text, i.e., sequences of characters. One character per line. Words separated by #.
  *
  * @param fn The path to the file.
  */
class TextStreamSourceCSV(fn: String) extends CSVStreamSource(fn) {

  override def line2Event(
                           line: Seq[String],
                           id: Int
                         ): GenericEvent = {
    val charNo = line(0).toInt
    val char = if (line(1).equalsIgnoreCase("#")) "SPACE" else line(1)
    val ge = GenericEvent(charNo, char, charNo)
    ge
  }
}
