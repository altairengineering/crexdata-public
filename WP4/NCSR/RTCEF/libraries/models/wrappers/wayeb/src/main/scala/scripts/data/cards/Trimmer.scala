package scripts.data.cards

import com.github.tototoshi.csv.{CSVReader, CSVWriter}

object Trimmer {
  def Run(
           inCsvFilePath: String,
           trimmedCsvFilePath: String,
           withReset: Boolean
         ): Unit = {
    val inputReader = CSVReader.open(inCsvFilePath)
    val writer = CSVWriter.open(trimmedCsvFilePath)
    val it = inputReader.iterator
    var lastSeen: Map[String, Trx] = Map.empty
    var lastTrxNo: Map[String, Int] = Map.empty
    var trxNo: Int = 0
    while (it.hasNext) {
      val line = it.next().toList
      val timestamp = line(0).toLong
      val amount = line(3).toDouble
      val pan = line(4)
      val isFraud = line(26).toInt
      val fraudType = line(27).toInt
      val newTrx = Trx(timestamp, amount, pan, isFraud, fraudType)
      var (amountDiff, timeDiff) =
        if (lastSeen.contains(pan)) {
          val prevTrx = lastSeen(pan)
          (newTrx.amount - prevTrx.amount, newTrx.timestamp - prevTrx.timestamp)
        }
        else {
          (0, -1)
        }
      trxNo += 1
      if (lastTrxNo.contains(pan)) {
        val prevTrxNo = lastTrxNo(pan)
        if (trxNo - prevTrxNo != 1) {
          val resetRow = List("-1", "-1", pan)
          if (withReset) writer.writeRow(resetRow)
          amountDiff = 0
          timeDiff = -1
        }
      }
      else {
        val resetRow = List("-1", "-1", pan)
        if (withReset) writer.writeRow(resetRow)
        amountDiff = 0
        timeDiff = -1
      }
      lastTrxNo += (pan -> trxNo)
      val row = List(trxNo.toString, timestamp.toString, pan, amount.toString, amountDiff.toString, timeDiff.toString, isFraud.toString, fraudType.toString)
      writer.writeRow(row)
      lastSeen += (pan -> newTrx)
    }
    inputReader.close()
    writer.close()

  }

}
