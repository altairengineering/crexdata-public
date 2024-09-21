package scripts.data.maritime

import com.github.tototoshi.csv.CSVWriter
import utils.SpatialUtils

import scala.io.{BufferedSource, Source}

object SyncedInterpolation {

  def RunInterpolation(
                        synopsesFilePath: String,
                        intervalSec: Int,
                        maxGap: Int,
                        speedThreshold: Double,
                        startTime: Long,
                        endTime: Long,
                        outCsvFilePath: String
                      ): Unit = {
    println("Reading file")
    val bufferedSource: BufferedSource = Source.fromFile(synopsesFilePath)
    val lines: List[String] = bufferedSource.getLines().toList
    bufferedSource.close()

    println("Converting to events")
    val critical: List[CriticalEvent] = lines.map(line => MaritimeUtils.str2CriticalEvent(line))
    val criticalSortedFiltered: List[CriticalEvent] = critical.filter(x => x.valid & x.timestamp >= startTime & x.timestamp <= endTime).sortBy(_.timestamp)

    val clockStart: Long = startTime //criticalSortedFiltered.head.timestamp
    val clockEnd: Long = endTime //criticalSortedFiltered.last.timestamp
    require(clockStart < clockEnd + intervalSec)
    var lastTimestamp: Map[Int, Long] = Map.empty
    val writer: CSVWriter = CSVWriter.open(outCsvFilePath)
    val (eventsAtStart, remainingAtStart): (List[CriticalEvent], List[CriticalEvent]) = getWindow(criticalSortedFiltered, clockStart, List.empty)
    var carriedEvents: Map[Int, (CriticalEvent, Int)] = eventsAtStart.map(x => (x.id, (x, 0))).toMap
    var remaining: List[CriticalEvent] = remainingAtStart
    var clock: Long = clockStart + intervalSec
    var percentRemaining = 100
    println("Starting sampling")
    while (clock <= clockEnd) {
      val window = getWindow(remaining, clock, List.empty)
      remaining = window._2
      val oldest = keepOldestTimestamp(window._1)
      val oldestProjected = oldest.mapValues(x => {
        val diffToClock = (clock - x.timestamp).toInt
        val (newLon, newLat) = SpatialUtils.projectPoint(x.lon, x.lat, diffToClock, x.speed, x.heading)
        val projected = CriticalEvent(x.id, clock, newLon, newLat, x.annotation, x.speed, x.heading, x.turn, x.course, x.valid)
        (projected, diffToClock)
      })
      val carriedToProject = carriedEvents.filter(x => (x._2._2 + intervalSec < maxGap) & (!oldestProjected.keySet.contains(x._1)))
      val carriedProjected = carriedToProject.mapValues(x => {
        val (newLon, newLat) = SpatialUtils.projectPoint(x._1.lon, x._1.lat, intervalSec, x._1.speed, x._1.heading)
        val projected = CriticalEvent(x._1.id, clock, newLon, newLat, x._1.annotation, x._1.speed, x._1.heading, x._1.turn, x._1.course, x._1.valid)
        (projected, x._2 + intervalSec)
      })
      carriedEvents = (oldestProjected ++ carriedProjected).filter(x => x._2._1.speed > speedThreshold)
      lastTimestamp = writeWithReset(carriedEvents.map(x => x._2._1).toList, lastTimestamp, maxGap, writer)
      clock += intervalSec
      val newPercentRemaining = (((clockEnd.toDouble - clock) / (clockEnd - clockStart)) * 100).toInt
      if (newPercentRemaining != percentRemaining) {
        percentRemaining = newPercentRemaining
        println("Remaining: " + percentRemaining + "%")
      }
    }
    writer.close()
    println("Sampling finished")

  }

  private def writeWithReset(
                              events: List[CriticalEvent],
                              lastTimestamp: Map[Int, Long],
                              maxGap: Int,
                              writer: CSVWriter
                            ): Map[Int, Long] = {
    var remaining = events
    var newLastTimestamp: Map[Int, Long] = lastTimestamp
    while (remaining.nonEmpty) {
      val thisEvent = remaining.head
      val id = thisEvent.id
      if (lastTimestamp.contains(id)) {
        if ((thisEvent.timestamp - lastTimestamp(id)) > maxGap) writer.writeRow(CriticalEvent(id).toRow)
      } else writer.writeRow(CriticalEvent(id).toRow)
      writer.writeRow(thisEvent.toRow)
      newLastTimestamp += (id -> thisEvent.timestamp)
      remaining = remaining.tail
    }
    newLastTimestamp
  }

  @scala.annotation.tailrec
  private def getWindow(
                         remaining: List[CriticalEvent],
                         windowEnd: Long,
                         window: List[CriticalEvent]
                       ): (List[CriticalEvent], List[CriticalEvent]) = {
    if (remaining.isEmpty) (window.reverse, remaining)
    else {
      val head = remaining.head
      if (head.timestamp <= windowEnd) {
        getWindow(remaining.tail, windowEnd, head :: window)
      } else (window.reverse, remaining)
    }
  }

  private def keepOldestTimestamp(window: List[CriticalEvent]): Map[Int, CriticalEvent] = {
    val grouped = window.groupBy(x => x.id)
    val sorted = grouped.mapValues(x => x.sortBy(_.timestamp))
    sorted.mapValues(x => x.last)
  }

}
