package stream.domain.archived

import java.io.File

import com.github.tototoshi.csv.CSVWriter
import com.typesafe.scalalogging.LazyLogging
import stream.GenericEvent
import stream.array.{EventStream, EventStreamI}

object MaritimePairedStream {
  def apply(
      filenameIn: String,
      filenameOut: String,
      sampleInterval: Int,
      convert: Boolean
  ): MaritimePairedStream = new MaritimePairedStream(filenameIn, filenameOut, sampleInterval, convert)

  def apply(
      filenameIn: String,
      filenameOut: String,
      sampleInterval: Int
  ): MaritimePairedStream = new MaritimePairedStream(filenameIn, filenameOut, sampleInterval, false)

  def apply(
      filenameIn: String,
      filenameOut: String
  ): MaritimePairedStream = new MaritimePairedStream(filenameIn, filenameOut, -1, false)

  def main(args: Array[String]): Unit = {
    val filenameIn = args(0)
    val sampleInterval = args(1).toInt
    val filenameOut = args(2)
    val mps = MaritimePairedStream(filenameIn, filenameOut, sampleInterval, true)
    val stream = mps.generateStream()
    //stream.writeCSV(filenameOut, List("Timestamp","EventType","pairId","lon1","lat1","speed1","heading1","lon2","lat2","speed2","heading2"))
  }
}

class MaritimePairedStream(
    filenameIn: String,
    filenameOut: String,
    sampleInterval: Int,
    convert: Boolean
) extends EventStreamI with LazyLogging {

  override def generateStream(): EventStream = {
    if (convert) convertAndGenerate()
    else readAndGenerate()
  }

  private def readAndGenerate(): EventStream = {
    val eventStream = new EventStream()
    var totalCounter = 1
    val bufferedSource = io.Source.fromFile(filenameIn)
    for (line <- bufferedSource.getLines) {
      val event = lineOfPair2Event(line, totalCounter)
      eventStream.addEvent(event)
      totalCounter += 1
    }
    eventStream
  }

  private def convertAndGenerate(): EventStream = {
    val eventStream = new EventStream()
    var totalCounter = 1
    var previousTimestamp: Long = -1
    var frame: Map[Int, GenericEvent] = Map.empty
    var pairs: Map[(Int, Int), Long] = Map.empty
    val bufferedSource = io.Source.fromFile(filenameIn)
    //val lines = bufferedSource.getLines
    //val linesNo = lines.size
    val startTime = 1443650461
    val endTime = 1459461541
    var lineNo = 0
    val fout = new File(filenameOut)
    val writer = CSVWriter.open(fout)
    val attributes = List("Timestamp", "EventType", "pairId", "lon1", "lat1", "speed1", "heading1", "lon2", "lat2", "speed2", "heading2")
    for (line <- bufferedSource.getLines) {
      lineNo += 1
      //logger.info("Progress " + lineNo / linesNo.toDouble)
      val event = line2Event(line)
      logger.info("Remaining " + (endTime - event.timestamp) / (endTime - startTime).toDouble)
      if (event.timestamp != previousTimestamp & previousTimestamp != -1) {
        val ids: Set[Int] = frame.keySet
        val idPermutations = utils.SetUtils.permutations(ids, 2).map(x => (x.head, x.tail.head)).filter(p => p._1 != p._2)
        val existingPairs = idPermutations & pairs.keySet
        val newPermutations = idPermutations &~ (existingPairs | existingPairs.map(p => (p._2, p._1)))
        var newPairs: Map[(Int, Int), Long] = Map.empty
        var newPermutationsToCheck = newPermutations
        while (newPermutationsToCheck.nonEmpty) {
          val newPerm = newPermutationsToCheck.head
          val newReversedPerm = (newPerm._2, newPerm._1)
          newPairs = newPairs + (newPerm -> frame(newPerm._1).timestamp)
          newPermutationsToCheck = newPermutationsToCheck - newPerm
          newPermutationsToCheck = newPermutationsToCheck - newReversedPerm
        }
        var pairsToAdd = newPairs.keySet
        while (pairsToAdd.nonEmpty) {
          val pairToAdd = pairsToAdd.head

          val previousPairTimestamp = if (pairs.contains(pairToAdd)) pairs(pairToAdd) else -1
          val currentPairTimestamp = frame(pairToAdd._1).timestamp
          pairs = pairs + (pairToAdd -> currentPairTimestamp)
          if (currentPairTimestamp - previousPairTimestamp != sampleInterval) {
            val resetEvent = pair2Event(frame(pairToAdd._1), frame(pairToAdd._2), totalCounter, "RESET")
            //eventStream.addEvent(resetEvent)
            writer.writeRow(getEventAsList(resetEvent, attributes, List.empty))
            totalCounter += 1
          }

          val pairedEvent = pair2Event(frame(pairToAdd._1), frame(pairToAdd._2), totalCounter, "SampledCriticalPair")
          //eventStream.addEvent(pairedEvent)
          writer.writeRow(getEventAsList(pairedEvent, attributes, List.empty))
          totalCounter += 1
          pairsToAdd = pairsToAdd - pairToAdd
        }
        pairsToAdd = existingPairs
        while (pairsToAdd.nonEmpty) {
          val pairToAdd = pairsToAdd.head
          val previousPairTimestamp = pairs(pairToAdd)
          val currentPairTimestamp = frame(pairToAdd._1).timestamp
          pairs = pairs + (pairToAdd -> currentPairTimestamp)
          if (currentPairTimestamp - previousPairTimestamp != sampleInterval) {
            val resetEvent = pair2Event(frame(pairToAdd._1), frame(pairToAdd._2), totalCounter, "RESET")
            //eventStream.addEvent(resetEvent)
            writer.writeRow(getEventAsList(resetEvent, attributes, List.empty))
            totalCounter += 1
          }
          val pairedEvent = pair2Event(frame(pairToAdd._1), frame(pairToAdd._2), totalCounter, "SampledCriticalPair")
          //eventStream.addEvent(pairedEvent)
          writer.writeRow(getEventAsList(pairedEvent, attributes, List.empty))
          totalCounter += 1
          pairsToAdd = pairsToAdd - pairToAdd
        }
        frame = Map(event.getValueOf("mmsi").toString.toInt -> event)
        previousTimestamp = event.timestamp
      } else {
        frame = frame + (event.getValueOf("mmsi").toString.toInt -> event)
        previousTimestamp = event.timestamp
      }
    }
    writer.close()
    eventStream
  }

  private def getEventAsList(e: GenericEvent, attributes: List[String], row: List[String]): List[String] = {
    attributes match {
      case Nil => row.reverse
      case head :: tail => getEventAsList(e, tail, e.getValueOf(head).toString :: row)
    }
  }

  private def lineOfPair2Event(
      line: String,
      id: Int
  ): GenericEvent = {
    val cols = line.split(",").map(_.trim)

    try {
      val timestamp = s"${cols(0)}".toLong
      val eventType = s"${cols(1)}"
      val pairId = s"${cols(2)}"

      val lon1 = s"${cols(3)}".toDouble
      val lat1 = s"${cols(4)}".toDouble
      val speed1 = s"${cols(5)}".toDouble
      val heading1 = s"${cols(6)}".toDouble

      val lon2 = s"${cols(7)}".toDouble
      val lat2 = s"${cols(8)}".toDouble
      val speed2 = s"${cols(9)}".toDouble
      val heading2 = s"${cols(10)}".toDouble

      val ge = GenericEvent(id, eventType, timestamp,
                            Map("pairId" -> pairId,
          "lon1" -> lon1,
          "lat1" -> lat1,
          "speed1" -> speed1,
          "heading1" -> heading1,
          "lon2" -> lon2,
          "lat2" -> lat2,
          "speed2" -> speed2,
          "heading2" -> heading2
        ))
      ge
    } catch {
      case e: Exception => {
        throw new Error("Could not parse line " + line)
      }
    }

  }

  private def pair2Event(
      event1: GenericEvent,
      event2: GenericEvent,
      id: Int,
      eventType: String
  ): GenericEvent = {
    val ge = GenericEvent(id, eventType, event1.timestamp,
                          Map("pairId" -> (event1.getValueOf("mmsi").toString + "-" + event2.getValueOf("mmsi")),
        "lon1" -> event1.getValueOf("lon"),
        "lat1" -> event1.getValueOf("lat"),
        "speed1" -> event1.getValueOf("speed"),
        "heading1" -> event1.getValueOf("heading"),
        "lon2" -> event2.getValueOf("lon"),
        "lat2" -> event2.getValueOf("lat"),
        "speed2" -> event2.getValueOf("speed"),
        "heading2" -> event2.getValueOf("heading")
      ))
    ge
  }

  private def line2Event(line: String): GenericEvent = {
    val cols = line.split(",").map(_.trim)
    try {
      val mmsi = s"${cols(1)}"
      val speed = s"${cols(4)}".toDouble
      val heading = s"${cols(5)}".toDouble
      val lon = s"${cols(2)}".toDouble
      val lat = s"${cols(3)}".toDouble
      val timestamp = s"${cols(0)}".toLong
      val ge = GenericEvent(0, "SampledCritical", timestamp,
                                                  Map("mmsi" -> mmsi, "speed" -> speed, "lon" -> lon, "lat" -> lat, "heading" -> heading))
      ge
    } catch {
      case e: Exception => {
        throw new Error("Could not parse line " + line)
      }
    }
  }

}
