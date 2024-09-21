package stream.domain.archived

import java.io.FileWriter

import model.markov.TransitionProbs
import stream.array.EventStream

import scala.collection.mutable.{Map, Set}
import scala.util.control.Breaks._

class FeedzaiStream(fileName: String, totalTrxs: Int, trainPercentage: Double, pattern: String) {
  require(totalTrxs > 0)
  require(trainPercentage > 0.0 & trainPercentage <= 0.5)
  private val patternsAllowed = Set("FarAwayLocations", "DecreasingAmounts", "IncreasingAmounts", "BigAfterSmall", "FlashAttack",
    "IncreasingAmountsExtra", "DecreasingAmountsExtra", "FlashAttackExtra")
  private val event2CharMap: Map[String, String] = Map("transaction" -> "t",
    "FarAwayLocation" -> "a",
    "BigAmount" -> "b",
    "SmallAmount" -> "s",
    "FlashAttack" -> "f",
    "DecreasingAmount" -> "d",
    "IncreasingAmount" -> "i"
  )
  require(patternsAllowed.contains(pattern))
  //private var charsAllowed:Set[Char] = Set('t','a','d','i')
  //private var pattern = "IncreasingAmounts"
  private val lastAmount = Map[String, Double]()
  private val lastAcquirerCountry = Map[String, Tuple2[Int, Long]]()
  private val lastTrxTimestamp = Map[String, Long]()
  private val firstTrxTimestamp = Map[String, Long]()
  private val eventStream = new EventStream()

  private val trainStream = new EventStream()
  private val testStream = new EventStream()
  private val trainTrxsNo = (totalTrxs * trainPercentage).toInt
  private val testTrxsNo = totalTrxs - trainTrxsNo

  private val probs = new TransitionProbs()

  private var farAwayWindow: Long = 300000
  private var big: Double = 1.0
  private var small: Double = 400.0
  private var bigAfterSmallWindow: Long = 300000
  private var flashAttackWindow: Long = 300000

  //generateStream()

  def generateStream(): Unit = {
    //val eventStream = new EventStream()
    var totalCounter = 0
    print("Creating train and test streams...")
    var trainCounter = 0
    var testCounter = 0
    val bufferedSource = io.Source.fromFile(fileName)
    var trainTypes: scala.collection.immutable.Set[String] = scala.collection.immutable.Set.empty[String]
    var testTypes: scala.collection.immutable.Set[String] = scala.collection.immutable.Set.empty[String]
    breakable {
      for (line <- bufferedSource.getLines) {
        val cols = line.split("\\|").map(_.trim)
        val pan = s"${cols(4)}"
        if (!pan.equalsIgnoreCase("card_pan")) {
          totalCounter += 1
          if (trainCounter <= trainTrxsNo) {
            trainCounter += 1
          } else if (testCounter <= testTrxsNo) {
            testCounter += 1
          } else {
            break
          }
          //println(totalCounter)
          val timestamp = s"${cols(0)}".toLong
          val amount = s"${cols(3)}".toDouble
          val acquirerCountry = s"${cols(10)}".toInt
          val isFraudStr = s"${cols(28)}"
          var isFraud = false
          if (isFraudStr.equalsIgnoreCase("1")) isFraud = true
          val eventType = determineEventType(pattern, timestamp, pan, amount, acquirerCountry)
          if (trainCounter <= trainTrxsNo) {
            val c = event2CharMap(eventType)
            val fe = new FeedzaiEvent(trainCounter, trainCounter, c, pan, isFraud)
            trainStream.addEvent(fe)
            trainTypes += c
          } else if (testCounter <= testTrxsNo) {
            val c = event2CharMap(eventType)
            val fe = new FeedzaiEvent(testCounter, testCounter, c, pan, isFraud)
            testStream.addEvent(fe)
            testTypes += c
          }
        }
      }
    }
    bufferedSource.close
    trainStream.setEventTypes(trainTypes)
    testStream.setEventTypes(testTypes)
    println("done. \nTrain/test size: " + trainStream.getSize + "/" + testStream.getSize)
    if (totalCounter < totalTrxs) {
      println("WARNING: file has less trxs than given")
    }
  }

  def setConfig(falw: Long, bg: Double, sm: Double, basw: Long, faw: Long): Unit = {
    farAwayWindow = falw
    big = bg
    small = sm
    bigAfterSmallWindow = basw
    flashAttackWindow = faw
    //charsAllowed = fzConfig.getEventCharsAllowed()
  }

  def determineEventType(pat: String, timestamp: Long, pan: String, amount: Double, acquirerCountry: Int): String = {
    var eventType = "transaction"

    if (pat.equalsIgnoreCase("IncreasingAmounts")) {
      if (!lastAmount.contains(pan)) {
        lastAmount += (pan -> amount)
      } else {
        val la = lastAmount(pan)
        if (amount > la) {
          eventType = "IncreasingAmount"
        }
        lastAmount(pan) = la
      }
    } else if (pat.equalsIgnoreCase("DecreasingAmounts")) {
      if (!lastAmount.contains(pan)) {
        lastAmount += (pan -> amount)
      } else {
        val la = lastAmount(pan)
        if (amount < la) {
          eventType = "DecreasingAmount"
        }
        lastAmount(pan) = la
      }
    } else if (pat.equalsIgnoreCase("FarAwayLocations")) {
      if (!lastAcquirerCountry.contains(pan)) {
        val ac: Tuple2[Int, Long] = (acquirerCountry, timestamp)
        lastAcquirerCountry += (pan -> ac)
      } else {
        val (lac, fts) = lastAcquirerCountry(pan)
        val timeSinceFirst = timestamp - fts
        if (acquirerCountry != lac & timeSinceFirst < farAwayWindow) {
          lastAcquirerCountry(pan) = (lac, fts)
          eventType = "FarAwayLocation"
        } else {
          lastAcquirerCountry(pan) = (lac, timestamp)
        }
      }
    } else if (pat.equalsIgnoreCase("BigAfterSmall")) {
      if (!lastTrxTimestamp.contains(pan)) {
        if (amount < small) {
          lastTrxTimestamp += (pan -> timestamp)
          eventType = "SmallAmount"
        }
      } else {
        val lastTs = lastTrxTimestamp(pan)
        val timeSinceSmall = timestamp - lastTs
        if (amount > big & timeSinceSmall < bigAfterSmallWindow) {
          eventType = "BigAmount"
          lastTrxTimestamp.remove(pan)
        }
      }

    } else if (pat.equalsIgnoreCase("FlashAttack")) {
      if (!firstTrxTimestamp.contains(pan)) {
        firstTrxTimestamp += (pan -> timestamp)
      } else {
        val firstTs = firstTrxTimestamp(pan)
        val timeSinceFirst = timestamp - firstTs
        if (timeSinceFirst < flashAttackWindow) {
          eventType = "FlashAttack"
        } else {
          firstTrxTimestamp += (pan -> timestamp)
        }
      }
    } else if (pat.equalsIgnoreCase("IncreasingAmountsExtra") | pat.equalsIgnoreCase("DecreasingAmountsExtra")) {
      if (!lastAmount.contains(pan)) {
        lastAmount += (pan -> amount)
      } else {
        val la = lastAmount(pan)
        if (amount > la) {
          eventType = "IncreasingAmount"
        } else if (amount < la) {
          eventType = "DecreasingAmount"
        }
        lastAmount(pan) = la
      }
    } else if (pat.equalsIgnoreCase("FlashAttackExtra")) {
      if (!lastAcquirerCountry.contains(pan)) {
        val ac: Tuple2[Int, Long] = (acquirerCountry, timestamp)
        lastAcquirerCountry += (pan -> ac)
      } else {
        val (lac, fts) = lastAcquirerCountry(pan)
        val timeSinceFirst = timestamp - fts
        if (acquirerCountry != lac & timeSinceFirst < farAwayWindow) {
          lastAcquirerCountry(pan) = (lac, fts)
          return "FarAwayLocation"
        } else {
          lastAcquirerCountry(pan) = (lac, timestamp)
        }
      }

      if (!firstTrxTimestamp.contains(pan)) {
        firstTrxTimestamp += (pan -> timestamp)
      } else {
        val firstTs = firstTrxTimestamp(pan)
        val timeSinceFirst = timestamp - firstTs
        if (timeSinceFirst < flashAttackWindow) {
          eventType = "FlashAttack"
        } else {
          firstTrxTimestamp += (pan -> timestamp)
        }
      }
    }

    eventType
  }

  def calculateProbs(order: Int): Unit = {
    print("Learning model @order " + order + "...")
    val t1 = System.nanoTime()
    val symbols = createSymbols()
    val labels = createLabels(symbols, order)
    val allCounters = Map[String, Counter]()
    val eventCounters = Map.empty[String, Int]
    for (i <- 0 to trainStream.getSize - 1) {
      val e = trainStream.getEvent(i)
      val pan = e.getValueOf("pan").toString
      val eventType = e.eventType
      if (!allCounters.contains(pan)) {
        val newCounter = new Counter(pan, order, labels, symbols)
        newCounter.updateCounters(eventType)
        allCounters += (pan -> newCounter)
      } else {
        val oldCounter = allCounters(pan)
        oldCounter.updateCounters(eventType)
      }
      if (!eventCounters.contains(eventType)) {
        eventCounters += (eventType -> 1)
      } else {
        eventCounters(eventType) += 1
      }
    }
    allCounters.retain((k, v) => v.getTotalCounter > 0)
    //println(allCounters)
    val mergedCounters = new Counter("global", order, labels, symbols)
    for ((k, v) <- allCounters) {
      mergedCounters.absorb(v)
    }
    //println(mergedCounters)

    val globalCounters = mergedCounters.getCounters
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
      for (label <- labels) {
        val thisCounters = globalCounters(label)
        var tc = 0
        for ((k, v) <- thisCounters) {
          tc += v
        }
        for ((k, v) <- thisCounters) {
          if (tc == 0) probs.addProb(label ::: List(k), 0.0)
          else probs.addProb(label ::: List(k), v.toDouble / tc.toDouble)
        }
      }
    }

    val totalEvents = eventCounters.foldLeft(0)(_ + _._2)
    for ((k, v) <- eventCounters) {
      val marginal = v.toDouble / totalEvents.toDouble
      probs.addMarginal(k, marginal)
    }

    val t2 = System.nanoTime()
    val t = t2 - t1
    println("done in " + (t / 1000000) + "ms.")
    println(probs)
  }

  def writeModel(fn: String): Unit = {
    val writer = new FileWriter(fn, true)
    writer.write(probs.toString + "\n")
    writer.close()
  }

  def createSymbols(): Set[String] = {
    val symbols = Set("t")
    if (pattern.equalsIgnoreCase("IncreasingAmounts")) {
      symbols += (event2CharMap("IncreasingAmount"))
    } else if (pattern.equalsIgnoreCase("DecreasingAmounts")) {
      symbols += (event2CharMap("DecreasingAmount"))
    } else if (pattern.equalsIgnoreCase("FarAwayLocations")) {
      symbols += (event2CharMap("FarAwayLocation"))
    } else if (pattern.equalsIgnoreCase("BigAfterSmall")) {
      symbols += (event2CharMap("BigAmount"), event2CharMap("SmallAmount"))
    } else if (pattern.equalsIgnoreCase("FlashAttack")) {
      symbols += (event2CharMap("FlashAttack"))
    } else if (pattern.equalsIgnoreCase("IncreasingAmountsExtra")) {
      symbols += (event2CharMap("IncreasingAmount"), event2CharMap("DecreasingAmount"))
    } else if (pattern.equalsIgnoreCase("DecreasingAmountsExtra")) {
      symbols += (event2CharMap("IncreasingAmount"), event2CharMap("DecreasingAmount"))
    } else if (pattern.equalsIgnoreCase("FlashAttackExtra")) {
      symbols += (event2CharMap("FlashAttack"), event2CharMap("FarAwayLocation"))
    }
    symbols
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
    var toRemove = Set.empty[List[String]]
    for (label <- labels) {
      if (label.size != k) toRemove += label //labels.remove(label)
    }
    for (tr <- toRemove) labels.remove(tr)
    labels
  }

  def getTrainStream: EventStream = trainStream
  def getTestStream: EventStream = testStream
  def getProbs: TransitionProbs = probs

}
