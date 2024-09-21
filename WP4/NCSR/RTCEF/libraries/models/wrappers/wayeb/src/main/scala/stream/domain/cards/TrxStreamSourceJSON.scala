package stream.domain.cards

import play.api.libs.json.{JsObject, Json}

import com.typesafe.scalalogging.LazyLogging

import stream.array.EventStream
import stream.source.EmitMode.EmitMode
import stream.source.{EmitMode, StreamSource}
import stream.{GenericEvent, ResetEvent}

object TrxStreamSourceJSON {
  /**
    * Constructor for transactions stream source.
    *
    * @param fn The path to the file.
    * @return The stream source.
    */
  def apply(fn: String): TrxStreamSourceJSON = new TrxStreamSourceJSON(fn)
}

/**
  * Stream source for credit card transactions given in a csv file.
  *
  * @param fn The path to the file.
  */
class TrxStreamSourceJSON(fn: String) extends StreamSource with LazyLogging {

  override protected def emitEvents(mode: EmitMode): EventStream = {
    val bufferedSource = io.Source.fromFile(fn)
    var totalCounter = 0L
    val eventStream = new EventStream()

    for (line <- bufferedSource.getLines) {
      val map = Json.parse(line).as[JsObject].value.toMap
      val timestamp = map("timestamp").toString.toLong
      val trxno = map("trx_no").toString.toLong
      val pan = map("pan").toString
      val amount = map("amount").toString.toDouble
      val amountDiff = map("amount_diff").toString.toDouble
      val timeDiff = map("time_diff").toString.toLong
      val isFraud = map("is_fraud").toString.toLong
      val fraudType = map("fraud_type").toString.toLong

      val arguments = Map(
        "time" -> timestamp,
        "pan" -> pan,
        "amount" -> amount,
        "amountDiff" -> amountDiff,
        "timeDiff" -> timeDiff,
        "isFraud" -> isFraud,
        "fraudType" -> fraudType,
        "nextCETimestamp" -> -1
      )

      send2Listeners(GenericEvent(totalCounter.toInt, "Trx", timestamp, arguments))
      totalCounter += 1

    }
    eventStream
  }
}
