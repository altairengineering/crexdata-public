package scripts.sessionsNtests.spsa.synthetic

import breeze.stats.distributions.Uniform
import com.typesafe.scalalogging.LazyLogging
import stream.array.EventStream
import stream.{GenericEvent, StreamFactory}
import model.vmm.Symbol
import model.vmm.pst.psa.{PSAUtils, ProbSuffixAutomaton}

import scala.collection.SortedMap

object StreamGenerator extends LazyLogging {
  def generatePSA(
      psaID: Int,
      trainEventsNo: Int,
      testEventsNo: Int,
      symbols: List[Symbol],
      maxOrder: Int,
      expansionProb: Double,
      resultsDir: String
  ): (ProbSuffixAutomaton, EventStream, EventStream) = {
    logger.info("Creating PSA with ID " + psaID)
    logger.info("Creating train stream of size " + trainEventsNo)
    var psa = PSAUtils.createPSA(symbols.toSet + Symbol(0), maxOrder, expansionProb)
    var (trainStream, percentByOrder) = StreamFactory.getStream(psa, trainEventsNo)
    var percentByOrderSorted = SortedMap(percentByOrder.toSeq: _*).toList.flatMap(e => List(e._1.toString, e._2.toString))
    var expectation = percentByOrder.map(p => p._1 * p._2).sum
    while (psa.maxOrder != maxOrder | expectation < maxOrder / 2) {
      logger.info("Trying again for PSA, expectation low " + expectation)
      psa = PSAUtils.createPSA(symbols.toSet + Symbol(0), maxOrder, expansionProb)
      val stream = StreamFactory.getStream(psa, trainEventsNo)
      trainStream = stream._1
      percentByOrder = stream._2
      percentByOrderSorted = SortedMap(percentByOrder.toSeq: _*).toList.flatMap(e => List(e._1.toString, e._2.toString))
      expectation = percentByOrder.map(p => p._1 * p._2).sum
    }
    val psaFn = resultsDir + "/psaOrder" + maxOrder + "Symbols" + symbols.length + "ID" + psaID + ".psa"
    //SerializationUtils.write2File[ProbSuffixAutomaton](List(psa),psaFn)
    logger.info("PSA generated\n " + psa.toString)
    logger.info("PSA size/maxOrder/ordersCounts: " + psa.size + "/" + psa.maxOrder + "/" + psa.statesByOrder)
    //val psaMatrix = psa.getTransitionMatrix
    //logger.info("PSA analysis:\n" + psaMatrix.toString)
    //val psaTrainStreamFn = resultsDir + "/streamTrain" + psaID + ".csv"
    //psaAnalysisWriter.writeRow(expectation.toString :: percentByOrderSorted)
    logger.info("Visits percentage by order/expectation: " + percentByOrder + "/" + expectation)

    logger.info("Creating test stream of size " + testEventsNo)
    val psaTestStreamFn = resultsDir + "/streamTestPSA" + psaID + "test" + 1 + ".csv"
    val (testStream, _) = StreamFactory.getStream(psa, testEventsNo)
    //testStream.writeCSV(psaTestStreamFn)
    //trainStream.writeCSV(psaTrainStreamFn)
    (psa, trainStream, testStream)
  }

  def generateStreamWithSpecificLtd(
      eventsNo: Int,
      order: Int,
      targetSymbol: String,
      defaultSymbol: String,
      allSymbols: Set[String],
      symbolOrLtdProb: Double,
      whichLtdProb: Double
  ): EventStream = {
    val ltdg = new LtdGenerator2(targetSymbol, order, allSymbols - defaultSymbol, whichLtdProb)
    generateStreamWithSpecificLtd(eventsNo, order, defaultSymbol, symbolOrLtdProb, ltdg)
  }

  def tagStreamAtDistance(
      es: EventStream,
      distance: Int
  ): EventStream = {
    val taggedStream = new EventStream()
    for (i <- 0 until es.getSize) {
      val event = es.getEvent(i)
      if (event.hasAttribute("distance") && event.getValueOfMap("distance").toString.toInt == distance) {
        val taggedEvent = GenericEvent(event.id, event.eventType, event.timestamp, Map("isEmitting" -> true))
        taggedStream.addEvent(taggedEvent)
      } else {
        val untaggedEvent = GenericEvent(event.id, event.eventType, event.timestamp, Map("isEmitting" -> false))
        taggedStream.addEvent(untaggedEvent)
      }
    }
    taggedStream
  }

  def generateStreamWithSpecificLtd(
      eventsNo: Int,
      order: Int,
      defaultSymbol: String,
      symbolOrLtdProb: Double,
      ltdg: LtdGenerator2
  ): EventStream = {
    val symbolOrLtdDist = Uniform(0, 1)
    var eventIndex = 0
    val eventStream = new EventStream()
    while (eventIndex <= eventsNo) {
      val symbolOrLtdSample = symbolOrLtdDist.sample()
      if (symbolOrLtdSample < symbolOrLtdProb) {
        eventIndex += 1
        val newEvent = GenericEvent(defaultSymbol, eventIndex)
        eventStream.addEvent(newEvent)
      } else {
        val ltd = ltdg.sampleLtdWithDistance.reverse
        for (i <- ltd.indices) {
          eventIndex += 1
          val newEvent = GenericEvent(eventIndex, ltd(i)._1, eventIndex, Map("distance" -> ltd(i)._2))
          eventStream.addEvent(newEvent)
        }
      }
    }
    eventStream
  }

  def generateTaggedStreamWithSpecificLtd(
      eventsNo: Int,
      order: Int,
      defaultSymbol: String,
      symbolOrLtdProb: Double,
      ltdg: LtdGenerator2,
      distance: Int
  ): EventStream = {
    require(distance > 0 & distance <= order)
    val symbolOrLtdDist = Uniform(0, 1)
    var eventIndex = 0
    val eventStream = new EventStream()
    while (eventIndex <= eventsNo) {
      val symbolOrLtdSample = symbolOrLtdDist.sample()
      if (symbolOrLtdSample < symbolOrLtdProb) {
        eventIndex += 1
        val newEvent = GenericEvent(eventIndex, defaultSymbol, eventIndex, Map("isEmitting" -> false))
        eventStream.addEvent(newEvent)
      } else {
        val ltd = ltdg.sampleLtdWithTag(distance).reverse
        for (i <- ltd.indices) {
          eventIndex += 1
          val isEmitting = ltd(i)._2
          val newEvent = GenericEvent(eventIndex, ltd(i)._1, eventIndex, Map("isEmitting" -> isEmitting))
          eventStream.addEvent(newEvent)
        }
      }
    }
    eventStream
  }

  def generateStreamWithOrderSpecificInjectedDependencies(
      eventsNo: Int,
      order: Int,
      defaultSymbol: String,
      middleSymbols: Set[String],
      symbolOrLtdProb: Double,
      whichLtdProb: Double
  ): EventStream = {
    //val ltdg = middleSymbols.map(s => LtdGenerator(order,whichLtdProb,"0","1",s)).toArray
    val ltdg = (2 to order).map(o => LtdGenerator(o, whichLtdProb, "0", "1", o.toString)).toArray
    val symbolOrLtdDist = Uniform(0, 1)
    //val whichLtdDist = Uniform(0,middleSymbols.size - 1)
    val whichLtdDist = Uniform(0, order - 1)

    var eventIndex = 0
    val eventStream = new EventStream()
    while (eventIndex <= eventsNo) {
      val symbolOrLtdSample = symbolOrLtdDist.sample()
      if (symbolOrLtdSample < symbolOrLtdProb) {
        eventIndex += 1
        val newEvent = GenericEvent(defaultSymbol, eventIndex)
        eventStream.addEvent(newEvent)
      } else {
        val whichLtdSample = whichLtdDist.sample().toInt
        val ltd = ltdg(whichLtdSample).sampleLtd
        for (i <- ltd.indices) {
          eventIndex += 1
          val newEvent = GenericEvent(ltd(i), eventIndex)
          eventStream.addEvent(newEvent)
        }

      }
    }
    eventStream
  }

  def generateStreamWithDummyInjectedDependencies(
      eventsNo: Int,
      order: Int,
      defaultSymbol: String,
      allSymbols: Set[String],
      target: String,
      symbolOrLtdProb: Double,
      whichLtdProb: Double
  ): EventStream = {
    val symbolOrLtdDist = Uniform(0, 1)
    val ltdg = new LtdGenerator1(order, allSymbols, target, whichLtdProb)
    var eventIndex = 0
    val eventStream = new EventStream()
    while (eventIndex <= eventsNo) {
      val symbolOrLtdSample = symbolOrLtdDist.sample()
      if (symbolOrLtdSample < symbolOrLtdProb) {
        eventIndex += 1
        val newEvent = GenericEvent(defaultSymbol, eventIndex)
        eventStream.addEvent(newEvent)
      } else {
        val ltd = ltdg.sampleLtd
        for (i <- ltd.indices) {
          eventIndex += 1
          val newEvent = GenericEvent(ltd(i), eventIndex)
          eventStream.addEvent(newEvent)
        }

      }
    }
    eventStream
  }
}
