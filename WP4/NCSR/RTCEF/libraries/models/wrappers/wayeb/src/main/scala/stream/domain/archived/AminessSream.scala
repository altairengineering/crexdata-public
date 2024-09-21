package stream.domain.archived

import model.markov.TransitionProbs
import stream.GenericEvent
import stream.array.{EventStream, EventStreamI}

import scala.collection.SortedMap
import scala.collection.mutable.{Map, Set}

class AminessSream(
    fn: String,
    vesselId: Int,
    start: Int,
    end: Int,
    o: Int
) extends EventStreamI {
  require(vesselId > 0)
  require(start > 0 & end > 0 & end > start)
  private val probs = new TransitionProbs()
  private var cumulativeProbs = Map.empty[List[String], SortedMap[Double, String]]
  private val event2CharMap: Map[String, String] = Map("turn" -> "t",
    "lowSpeedStart" -> "l",
    "lowSpeedEnd" -> "s",
    "gapStart" -> "g",
    "gapEnd" -> "p",
    "speedChange" -> "c"
  )

  private val event2CharMapExt: Map[String, String] = Map(
    "turnNorthNormal" -> "A",
    "turnEastNormal" -> "B",
    "turnSouthNormal" -> "C",
    "turnWestNormal" -> "D",
    "lowSpeedStartNorthNormal" -> "E",
    "lowSpeedStartEastNormal" -> "F",
    "lowSpeedStartSouthNormal" -> "G",
    "lowSpeedStartWestNormal" -> "H",
    "lowSpeedEndNorthNormal" -> "I",
    "lowSpeedEndEastNormal" -> "J",
    "lowSpeedEndSouthNormal" -> "K",
    "lowSpeedEndWestNormal" -> "L",
    "gapStartNorthNormal" -> "M",
    "gapStartEastNormal" -> "N",
    "gapStartSouthNormal" -> "O",
    "gapStartWestNormal" -> "P",
    "gapEndNorthNormal" -> "Q",
    "gapEndEastNormal" -> "R",
    "gapEndSouthNormal" -> "S",
    "gapEndWestNormal" -> "T",
    "speedChangeNorthNormal" -> "U",
    "speedChangeEastNormal" -> "V",
    "speedChangeSouthNormal" -> "W",
    "speedChangeWestNormal" -> "X" /*,

    "turnNorthHigh" -> "A",
    "turnEastHigh" -> "B",
    "turnSouthHigh" -> "C",
    "turnWestHigh" -> "D",
    "lowSpeedStartNorthHigh" ->"E",
    "lowSpeedStartEastHigh" ->"F",
    "lowSpeedStartSouthHigh" ->"G",
    "lowSpeedStartWestHigh" ->"H",
    "lowSpeedEndNorthHigh" ->"I",
    "lowSpeedEndEastHigh" ->"J",
    "lowSpeedEndSouthHigh" ->"K",
    "lowSpeedEndWestHigh" ->"L",
    "gapStartNorthHigh" -> "M",
    "gapStartEastHigh" -> "N",
    "gapStartSouthHigh" -> "O",
    "gapStartWestHigh" -> "P",
    "gapEndNorthHigh" -> "Q",
    "gapEndEastHigh" -> "R",
    "gapEndSouthHigh" -> "S",
    "gapEndWestHigh" -> "T",
    "speedChangeNorthHigh" -> "U",
    "speedChangeEastHigh" -> "V",
    "speedChangeSouthHigh" -> "W",
    "speedChangeWestHigh" -> "X"*/ )

  private var activity = Map.empty[Int, Int]
  private val order = o
  private var globalCounters = Map.empty[List[String], Map[String, Int]]
  private var globalLabels = Set.empty[List[String]]
  private var eventCounters = Map.empty[String, Int]

  def generateStream(): EventStream = {
    val maxGap = 7200
    val eventStream = new EventStream()
    var counters = Map.empty[String, Int]
    for ((k, v) <- event2CharMapExt) {
      counters += (v -> 0)
    }
    //counters += ("e" -> 0)
    initCounters()
    var totalCounter = 0
    var segmentCounter = 0
    val segCounters = Set.empty[Int]
    var aminEvent: String = null
    var vid: Int = 0
    var timestamp: Int = 0
    var lastTimestamp: Int = 0
    //var longestGap = 0
    //var shortestGap = 0
    var lastCritical = ""
    var heading = 0.0
    var speed = 0.0
    println("Reading Aminess stream...")
    val bufferedSource = io.Source.fromFile(fn)
    for (line <- bufferedSource.getLines) {
      //println(line.split("|"))
      val line2 = line.replace("|", ",")
      val cols = line2.split(",").map(_.trim)
      aminEvent = s"${cols(0)}"
      if (event2CharMap.contains(aminEvent)) {
        timestamp = s"${cols(1)}".toInt
        vid = s"${cols(3)}".toInt

        //if (activity.contains(vid)) activity(vid) += 1
        //else activity += (vid -> 1)

        if (vid == vesselId & timestamp == lastTimestamp) {
          //println("FOUND EVENT AT SAME TIMESTAMP: " + vid + "/" + aminEvent + "/" + timestamp)
        }
        if (vid == vesselId & timestamp >= start & timestamp <= end & timestamp > lastTimestamp) {
          val gap = timestamp - lastTimestamp
          if (gap > maxGap) {
            //println("\n\tGAP LONGER THAN " + maxGap + "\n\tPREVIOUS SEGMENT COUNTER: " + segmentCounter)
            segCounters += segmentCounter
            segmentCounter = 0
          }
          /*if (lastTimestamp!=0 & gap>1) {
            if (gap<=maxGap) {
              addEmptyEvents(eventStream, lastTimestamp, timestamp)
              counters("e") += gap - 1
              totalCounter += gap - 1
            }
            if (gap>longestGap) longestGap = gap
            if (shortestGap==0) shortestGap = gap
            else if (gap<shortestGap) shortestGap = gap
          }*/
          lastCritical = aminEvent

          /*totalCounter += 1
          segmentCounter += 1
          println(aminEvent + "," + vid + "," + timestamp + "," + gap)
          val c = event2CharMap(aminEvent)
          updateCounters(eventStream,c,totalCounter)
          eventStream.addEvent(new Event(totalCounter, totalCounter, c))
          counters(c) += 1
          lastTimestamp = timestamp*/
        } else {
          lastCritical = ""
        }
      } else if (aminEvent.equalsIgnoreCase("velocity")) {
        timestamp = s"${cols(1)}".toInt
        speed = s"${cols(4)}".toDouble
        heading = s"${cols(5)}".toDouble
        val eventExt = event2Ext(lastCritical, speed, heading)
        if (!eventExt.equalsIgnoreCase("")) {
          val c = event2CharMapExt(eventExt)
          totalCounter += 1
          segmentCounter += 1
          //println(eventExt + "," + vid + "," + timestamp + "," + speed + "," + heading)
          updateCounters(eventStream, c, totalCounter)
          eventStream.addEvent(GenericEvent(totalCounter, c.toString, timestamp,
                                            scala.collection.immutable.Map("vid" -> vid, "speed" -> speed, "heading" -> heading)))
          counters(c) += 1
          lastTimestamp = timestamp
        }
      }
    }
    bufferedSource.close
    //calculateProbs(counters, totalCounter)
    calculateProbs()
    println(counters)
    println(totalCounter)
    val filteredSegCounters = segCounters.filter(_ > 600)
    println(segCounters.size + ":" + segCounters)
    println(filteredSegCounters.size + ":" + filteredSegCounters)
    //println("lgap/sgap:" + longestGap + "/" + shortestGap)
    eventStream.setProbs(probs)
    //eventStream.setCumulativeProbs(cumulativeProbs)
    println(probs)
    //getMostActiveVessels(10)
    eventStream.setEventTypes(event2CharMapExt.values.toSet)
    eventStream
  }

  private def event2Ext(ls: String, speed: Double, heading: Double): String = {
    if (ls.equalsIgnoreCase("")) return ""
    var ext = ""

    var headingStr = ""
    if (heading >= 315.0 | heading < 45.0) headingStr = "North"
    else if (heading >= 45.0 & heading < 135.0) headingStr = "East"
    else if (heading >= 135.0 & heading < 225.0) headingStr = "South"
    else if (heading >= 225.0 & heading < 315.0) headingStr = "West"
    else println(heading)

    var speedStr = "Normal"
    //if (speed>30.0) speedStr="High"

    ext = ls + headingStr + speedStr
    ext
  }
  /*
  private def calculateProbs(cnts: Map[Char,Int], tc: Int): Unit = {
    var cumProb = 0.0
    for ((k,v) <- cnts) {
      val p = v.toDouble/tc.toDouble
      probs += (k -> p)
    }
    for ((k,v) <- probs) {
      cumProb += v
      cumulativeProbs += (k -> cumProb)
    }
  }
*/

  private def calculateProbs(): Unit = {
    val totalEvents = eventCounters.foldLeft(0)(_ + _._2)
    for ((k, v) <- eventCounters) {
      val marginal = v.toDouble / totalEvents.toDouble
      probs.addMarginal(k, marginal)
    }
    if (order == 0) {
      val thisCounters = globalCounters(List.empty[String])
      var tc = 0
      for ((k, v) <- thisCounters) {
        tc += v
      }
      for ((k, v) <- thisCounters) {
        if (tc == 0) probs.addProb(List(k), 0.0)
        else probs.addProb(List(k), v.toDouble / tc.toDouble)
      }
    } else {
      for (label <- globalLabels) {
        val thisCounters = globalCounters(label)
        var tc = 0
        for ((k, v) <- thisCounters) {
          tc += v
        }
        for ((k, v) <- thisCounters) {
          var marginal = 0.0
          if (tc != 0.0) {
            marginal = v.toDouble / tc.toDouble
          }

          if (tc == 0) probs.addProb(label ::: List(k), 0.0)
          else probs.addProb(label ::: List(k), v.toDouble / tc.toDouble)
        }
      }
    }
  }

  private def updateCounters(es: EventStream, c: String, i: Int): Unit = {
    if (i > order) {
      var label = List.empty[String]
      if (order == 0) label = List.empty[String]
      else label = getLastEventsAsLabel(es, i)
      val thisCounters = globalCounters(label)
      thisCounters(c) += 1
      eventCounters(c) += 1
    }
  }

  private def initCounters(): Unit = {
    val symbols = Set.empty[String]
    for ((k, v) <- event2CharMapExt) {
      symbols += v
    }
    for (s <- symbols) {
      eventCounters += (s -> 0)
    }
    //var labels = Set[String]()
    if (order == 0) {
      globalLabels = Set(List.empty[String])
    } else {
      globalLabels = createLabels(symbols, order)
    }
    for (label <- globalLabels) {
      val thisCounters = Map.empty[String, Int]
      for (symbol <- symbols) {
        thisCounters += (symbol -> 0)
      }
      globalCounters += (label -> thisCounters)
    }
  }

  private def getLastEventsAsLabel(es: EventStream, i: Int): List[String] = {
    var label = List.empty[String]
    for (j <- i - order - 1 to i - 2) {
      label = label ::: List(es.getEvent(j).eventType)
    }
    label
  }

  private def addEmptyEvents(es: EventStream, lt: Int, nt: Int): Unit = {
    for (t <- lt + 1 to nt - 1) {
      es.addEvent(GenericEvent("e", t.toLong))
    }
  }

  private def getMostActiveVessels(k: Int): Unit = {
    val activitySorted = activity.toSeq.sortBy(_._2)
    for (i <- activitySorted.size - 1 - k to activitySorted.size - 1) {
      println("Active vessel: " + activitySorted(i))
    }
  }

  def createLabels(symbols: Set[String], k: Int): Set[List[String]] = {
    var labels = Set.empty[List[String]]
    labels += List.empty[String]

    var labelsK = Set.empty[List[String]]
    for (i <- 1 to k) {
      labelsK.clear()
      for (p <- labels) {
        for (e <- symbols) {
          val newp = p ::: List(e)
          labelsK += newp
        }
      }
      labels ++= labelsK
    }
    for (label <- labels) {
      if (label.size != k) labels.remove(label)
    }
    labels
  }

}
