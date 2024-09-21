package stream.domain.cards

import stream.{GenericEvent, ResetEvent}
import stream.source.CSVStreamSource

object TrxStreamSourceCSV {
  /**
    * Constructor for transactions stream source.
    *
    * @param fn The path to the file.
    * @return The stream source.
    */
  def apply(fn: String): TrxStreamSourceCSV = new TrxStreamSourceCSV(fn)
}

/**
  * Stream source for credit card transactions given in a csv file.
  *
  * @param fn The path to the file.
  */
class TrxStreamSourceCSV(fn: String) extends CSVStreamSource(fn) {

  override def line2Event(
                           line: Seq[String],
                           id: Int
                         ): GenericEvent = {
    val trxNo = line(0).toInt
    val pan = line(2)
    if (trxNo == -1) {
      ResetEvent(Map("pan" -> pan))
    } else {
      val timestamp = line(1).toLong
      val amount = line(3).toDouble
      val amountDiff = line(4).toDouble
      val timeDiff = line(5).toLong
      val isFraud = line(6).toInt
      val fraudType = line(7).toInt
      val nextCETimestamp = if (line.size > 8) line(8).toInt else -1
      val ge = GenericEvent(id, "Trx", trxNo,
                            Map("time" -> timestamp, "pan" -> pan, "amount" -> amount, "amountDiff" -> amountDiff, "timeDiff" -> timeDiff,
          "isFraud" -> isFraud, "fraudType" -> fraudType, "nextCETimestamp" -> nextCETimestamp))
      ge
    }
  }
}
