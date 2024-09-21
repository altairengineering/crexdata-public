package scripts.sessionsNtests.spsa.synthetic

import breeze.stats.distributions.Uniform
import com.github.tototoshi.csv.CSVWriter
import com.typesafe.scalalogging.LazyLogging
import fsm.symbolic.sre.SREUtils
import fsm.symbolic.sfa.SFAUtils
import fsm.symbolic.sfa.logic.{AtomicSentence, PredicateConstructor, Sentence}
import fsm.symbolic.sfa.sdfa.SDFAUtils
import fsm.symbolic.sfa.snfa.SNFAUtils
import fsm.classical.pattern.regexp.SymbolNode
import profiler.WtProfiler
import stream.source.ArrayStreamSource
import stream.GenericEvent
import stream.array.EventStream
import model.vmm.pst.spsa.SymbolicPSA
import model.vmm.{Isomorphism, Symbol, VMMUtils}
import ui.ConfigUtils
import workflow.provider.source._
import workflow.provider._
import workflow.provider.source.matrix.{MCSourceDirect, MCSourceSPSA}
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.source.sdfa.{SDFASourceDirect, SDFASourceFormula}
import workflow.provider.source.spsa.SPSASourceDirect
import workflow.provider.source.wt.WtSourceMatrix
import workflow.task.engineTask.ERFTask
import workflow.task.estimatorTask.MatrixMLETask

object RunComparisonInjected extends LazyLogging {
  def main(args: Array[String]): Unit = {
    val resultsDir = ConfigInjected.resultsDir
    val resultsFile = ConfigInjected.resultsFile
    val maxOrder = ConfigInjected.maxOrder
    val disMaxOrder = ConfigInjected.disMaxOrder
    val symbolsNo = ConfigInjected.symbolsNo
    val trainEventsNo = ConfigInjected.trainEventsNo
    val testEventsNo = ConfigInjected.testEventsNo
    val testsNo = ConfigInjected.testsNo
    val horizon = ConfigInjected.horizon
    val maxSpread = ConfigInjected.maxSpread
    val predictionThreshold = ConfigInjected.predictionThreshold

    val symbols = (1 to symbolsNo).map(i => Symbol(i)).toList
    val patternPredicates = PredicateConstructor.getEventTypePred("1") // symbols.map(s => PredicateConstructor.getEventTypePred(s.value.toString))
    val extraPredicates = symbols.tail.map(s => PredicateConstructor.getEventTypePred(s.value.toString))
    val allPredicates = Set(patternPredicates) ++ extraPredicates.toSet
    val exclusives = Set(allPredicates)
    val extras = extraPredicates.map(ep => AtomicSentence(ep).asInstanceOf[Sentence]).toSet

    //val patternsSingleSymbol = symbols.map(s => SymbolNode(s.value.toString))
    //val patternAC = OperatorNode(CONCAT,List(SymbolNode("A"),SymbolNode("C")))
    val policy = ConfigUtils.defaultPolicy
    //val formulasSingleSymbol = patternsSingleSymbol.map(p => CEPLUtils.re2cepl(p))

    val pattern1 = SymbolNode("1")
    //val pattern2 = SymbolNode("2")

    val patterns = List(pattern1)
    val formulas = patterns.map(p => SREUtils.re2formula(p))

    val formulasArray = formulas.toArray

    val writer = CSVWriter.open(resultsFile, append = true)

    for (test <- 1 to testsNo) {
      logger.info("Creating train stream of size " + trainEventsNo)
      //val trainStream = generateStreamWithInjectedDependencies(trainEventsNo)
      val trainStream = generateStreamWithMultipleInjectedDependencies(trainEventsNo, maxOrder, symbolsNo.toString)
      val psaTrainStreamFn = resultsDir + "/streamTrain" + test + ".csv"
      //trainStream.writeCSV(psaTrainStreamFn)
      logger.info("Creating test stream of size " + testEventsNo)
      val testStream = generateStreamWithMultipleInjectedDependencies(testEventsNo, maxOrder, symbolsNo.toString) //generateStreamWithInjectedDependencies(testEventsNo)
      val psaTestStreamFn = resultsDir + "/streamTest" + test + ".csv"
      //testStream.writeCSV(psaTestStreamFn)

      for (f <- formulasArray.indices) {
        val formula = formulasArray(f)
        val snfaStream = SNFAUtils.buildSNFAForStream(formula)
        val sdfa = SFAUtils.determinizeI(snfaStream, exclusives, extras)
        SDFAUtils.checkForDead(sdfa)
        val sdfaPol = SDFAUtils.setPolicy(sdfa, policy)
        val sentences = sdfaPol.getSentences
        val minTermsWithoutNeg = symbols.map(symbol => sentences.filter(sentence => sentence.evaluate(GenericEvent(symbol.value.toString, 1))).head)
        val negationSentence = sentences.filter(sentence => !symbols.exists(symbol => sentence.evaluate(GenericEvent(symbol.value.toString, 1)))).head
        val minTerms = negationSentence :: minTermsWithoutNeg
        val isoSymbols = Symbol(0) :: symbols
        val iso = Isomorphism(minTerms, isoSymbols)

        var row = List(test.toString)
        for (dm <- disMaxOrder) {
          logger.info("Learning SPSA@" + dm.toString)
          val maxNoStates = ConfigInjected.maxNoStates
          val (spsaLearnt, part, psaLearnt) = VMMUtils.learnSPSAFromSingleFormula((formula, dm, "$"), exclusives, extras, policy, ArrayStreamSource(trainStream), maxNoStates, iso)
          logger.info("PSA learnt\n " + psaLearnt.toString)
          logger.info("PSA size/maxOrder/ordersCounts: " + psaLearnt.size + "/" + psaLearnt.maxOrder + "/" + psaLearnt.statesByOrder)

          logger.info("Running forecasting with learnt")
          val profilerLearnt = runFore(spsaLearnt, testStream, horizon, predictionThreshold, maxSpread)
          val (precLearnt, spreadLearnt, intervalScoreLearnt) = (profilerLearnt.getStatFor("precision", 0), profilerLearnt.getStatFor("spread", 0), profilerLearnt.getStatFor("interval", 0))
          row = row ::: List(dm.toString, spsaLearnt.getSize.toString, precLearnt.toString, spreadLearnt.toString, intervalScoreLearnt.toString)

          logger.info("Creating disambiguated SDFA")
          //val disOrder = getDisOrder
          val formulasForDis = List((formula, dm, "$"))
          val sdfaDisProvF = SDFAProvider(SDFASourceFormula(formulasForDis, policy, exclusives, extras))
          val sdfaDis = sdfaDisProvF.provide()
          val sdfaDisProv = SDFAProvider(SDFASourceDirect(sdfaDis.map(el => el.sdfa)))
          val fsmDisProv = FSMProvider(sdfaDisProv)
          val met = MatrixMLETask(fsmDisProv, ArrayStreamSource(trainStream))
          val mc = met.execute()._1
          val mcProv = MarkovChainProvider(MCSourceDirect(mc))
          val wtdProv = WtProvider(WtSourceMatrix(fsmDisProv, mcProv, horizon, false))
          val predProv = ForecasterProvider(ForecasterSourceBuild(fsmDisProv, wtdProv, horizon, predictionThreshold, maxSpread))

          logger.info("Running forecasting with disambiguated SDFA")
          val erfDisTask = ERFTask(fsmDisProv, predProv, ArrayStreamSource(testStream), show = true)
          val profilerDis = erfDisTask.execute()
          profilerDis.printProfileInfo()
          val (precDis, spreadDis, intervalScoreDis) = (profilerDis.getStatFor("precision", 0), profilerDis.getStatFor("spread", 0), profilerDis.getStatFor("interval", 0))
          row = row ::: List(dm.toString, sdfaDis.head.size.toString, precDis.toString, spreadDis.toString, intervalScoreDis.toString)

          logger.info("DIS@" + dm + "\n Precision/Spread/Interval + \n" + precDis + "\n" + spreadDis + "\n" + intervalScoreDis)
          logger.info("LEARNT@" + dm + "\n Precision/Spread/Interval + \n" + precLearnt + "\n" + spreadLearnt + "\n" + intervalScoreLearnt)

        }
        writer.writeRow(row)

      }
    }
    writer.close()

  }

  def runFore(
      spsa: SymbolicPSA,
      stream: EventStream,
      horizon: Int,
      predictionThreshold: Double,
      maxSpread: Int
  ): WtProfiler = {
    val spsaProv = SPSAProvider(SPSASourceDirect(List(spsa)))
    val fsmProv = FSMProvider(spsaProv)
    val matrixProv = MarkovChainProvider(MCSourceSPSA(fsmProv))
    val wtdProv = WtProvider(WtSourceMatrix(fsmProv, matrixProv, horizon, false))
    val predProv = ForecasterProvider(ForecasterSourceBuild(fsmProv, wtdProv, horizon, predictionThreshold, maxSpread))
    val erf = ERFTask(fsmProv, predProv, ArrayStreamSource(stream), show = true)
    val profiler = erf.execute()
    profiler.printProfileInfo()
    profiler
  }

  def generateStreamWithMultipleInjectedDependencies(
      eventsNo: Int,
      maxOrder: Int,
      defaultSymbol: String
  ): EventStream = {
    val whichLtdProb = ConfigInjected.whichLtdProb
    val ltdg = (2 to maxOrder).map(o => LtdGenerator(o, whichLtdProb)).toArray
    val symbolOrLtdProb = ConfigInjected.symbolOrLtdProb

    val symbolOrLtdDist = Uniform(0, 1)
    val symbolDist = Uniform(0, 1)
    val whichLtdDist = Uniform(0, maxOrder - 1)

    var eventIndex = 0
    val eventStream = new EventStream()
    while (eventIndex <= eventsNo) {
      val symbolOrLtdSample = symbolOrLtdDist.sample()
      if (symbolOrLtdSample < symbolOrLtdProb) {
        eventIndex += 1
        val symbolSample = symbolDist.sample()
        val newEvent = GenericEvent(defaultSymbol, eventIndex) //if (symbolSample < whichSymbolProb) GenericEvent(symbol0, eventIndex)
        //else GenericEvent(symbol1, eventIndex)
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

  def generateStreamWithInjectedDependencies(eventsNo: Int): EventStream = {
    val symbol0 = "0"
    val symbol1 = "1"
    val symbol2 = "2"
    /*val ltd1 = List(
      symbol1, symbol1, symbol1, symbol1, symbol1,
      //symbol1, symbol1, symbol1, symbol1, symbol1,
      //symbol1, symbol1, symbol1, symbol1, symbol1,
      symbol1, symbol1, symbol1, symbol1, symbol0).toArray
    val ltd2 = List(
      symbol0, symbol1, symbol1, symbol1, symbol1,
      //symbol1, symbol1, symbol1, symbol1, symbol1,
      //symbol1, symbol1, symbol1, symbol1, symbol1,
      symbol1, symbol1, symbol1, symbol1, symbol1).toArray*/

    val ltd1 = List(symbol2, symbol2, symbol0).toArray
    val ltd2 = List(symbol0, symbol2, symbol1).toArray
    val symbolOrLtdProb = 0.95
    val whichLtdProb = 0.5
    val whichSymbolProb = 0.99

    val symbolOrLtdDist = Uniform(0, 1)
    val symbolDist = Uniform(0, 1)
    val whichLtdDist = Uniform(0, 1)

    var eventIndex = 0
    val eventStream = new EventStream()
    while (eventIndex <= eventsNo) {
      val symbolOrLtdSample = symbolOrLtdDist.sample()
      if (symbolOrLtdSample < symbolOrLtdProb) {
        eventIndex += 1
        val symbolSample = symbolDist.sample()
        val newEvent = GenericEvent(symbol0, eventIndex) //if (symbolSample < whichSymbolProb) GenericEvent(symbol0, eventIndex)
        //else GenericEvent(symbol1, eventIndex)
        eventStream.addEvent(newEvent)
      } else {
        val whichLtdSample = whichLtdDist.sample()
        val ltd = if (whichLtdSample < whichLtdProb) ltd1 else ltd2 //if (whichLtdSample < whichLtdProb(1)) ltd2 else ltd3
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
