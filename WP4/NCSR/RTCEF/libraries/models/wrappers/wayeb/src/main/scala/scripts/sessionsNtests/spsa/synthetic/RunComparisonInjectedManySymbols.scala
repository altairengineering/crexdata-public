package scripts.sessionsNtests.spsa.synthetic

import com.github.tototoshi.csv.CSVWriter
import com.typesafe.scalalogging.LazyLogging
import fsm.SPSAInterface
import fsm.symbolic.sre.BooleanOperator
import fsm.symbolic.sfa.logic._
import fsm.classical.pattern.regexp.SymbolNode
import profiler.WtProfiler
import stream.array.EventStream
import stream.source.ArrayStreamSource
import model.vmm.pst.spsa.SymbolicPSA
import model.vmm.{Isomorphism, Symbol, VMMUtils}
import ui.ConfigUtils
import workflow.provider.source._
import workflow.provider._
import workflow.provider.source.dfa.DFASourceRegExp
import workflow.provider.source.matrix.{MCSourceDirect, MCSourceSPSA}
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.source.sdfa.{SDFASourceDFA, SDFASourceDirect}
import workflow.provider.source.spsa.SPSASourceDirect
import workflow.provider.source.wt.WtSourceMatrix
import workflow.task.engineTask.ERFTask
import workflow.task.estimatorTask.MatrixMLETask

object RunComparisonInjectedManySymbols extends LazyLogging {
  def main(args: Array[String]): Unit = {
    logger.info("Running ComparisonInjectedManySymbols with configuration " + ConfigInjectedManySymbols.toString)
    logger.info("Creating isomorphism")
    val symbols = (0 until ConfigInjectedManySymbols.symbolsNo).map(i => Symbol(i)).toList
    val simpleSentences = symbols.tail.map(s => AtomicSentence(PredicateConstructor.getEventTypePred(s.value.toString)).asInstanceOf[Sentence])
    val negatedSentences = simpleSentences.map(s => ComplexSentence(BooleanOperator.NOT, List(s)).asInstanceOf[Sentence]).toSet
    val negationsSentence = LogicUtils.connectWithAND(negatedSentences)
    val sentences = negationsSentence :: simpleSentences
    val iso = Isomorphism(sentences, symbols)

    logger.info("Building SDFA@0")
    logger.info("Building DFA")
    val pattern = SymbolNode(ConfigInjectedManySymbols.target)
    val dfaProvider0 = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, 0, symbols.map(s => s.toString).toSet))
    val dfa0 = dfaProvider0.provide().head.dfa
    logger.info("Using isomorphism to build SDFA")
    val sdfaProvider0 = SDFAProvider(SDFASourceDFA(dfa0, iso))
    val sdfa0 = sdfaProvider0.provide().head.sdfa
    //logger.info("SDFA@0\n" + sdfa0.toString)

    logger.info("Building SDFA@" + ConfigInjectedManySymbols.disMaxOrder)
    logger.info("Building DFA")
    val dfaProviderm = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, ConfigInjectedManySymbols.disMaxOrder, symbols.map(s => s.toString).toSet))
    val dfam = dfaProviderm.provide().head.dfa
    logger.info("Using isomorphism to build SDFA")
    val sdfaProviderm = SDFAProvider(SDFASourceDFA(dfam, iso))
    val sdfam = sdfaProviderm.provide().head.sdfa
    //logger.info("SDFA@" + ConfigInjectedManySymbols.disMaxOrder + "\n" + sdfa0.toString)

    val writer = CSVWriter.open(ConfigInjectedManySymbols.resultsFile, append = true)
    for (test <- 1 to ConfigInjectedManySymbols.testsNo; maxSpread <- ConfigInjectedManySymbols.maxSpread; predictionThreshold <- ConfigInjectedManySymbols.predictionThreshold) {

      logger.info("Creating train stream of size " + ConfigInjectedManySymbols.trainEventsNo)
      /*val middleSymbols = (2 until ConfigInjectedManySymbols.symbolsNo-1).map(s => s.toString).toSet
      val trainStream = generateStreamWithOrderSpecificInjectedDependencies(
        ConfigInjectedManySymbols.trainEventsNo,
        ConfigInjectedManySymbols.maxOrder,
        (ConfigInjectedManySymbols.symbolsNo-1).toString,
        middleSymbols
      )*/
      val trainStream = StreamGenerator.generateStreamWithDummyInjectedDependencies(
        ConfigInjectedManySymbols.trainEventsNo,
        ConfigInjectedManySymbols.maxOrder,
        (ConfigInjectedManySymbols.symbolsNo - 1).toString,
        symbols.map(s => s.toString).toSet,
        ConfigInjectedManySymbols.target,
        ConfigInjectedManySymbols.symbolOrLtdProb,
        ConfigInjectedManySymbols.whichLtdProb
      )
      //val psaTrainStreamFn = ConfigInjectedManySymbols.resultsDir + "/streamTrain" + test + ".csv"
      //trainStream.writeCSV(psaTrainStreamFn)
      logger.info("Creating test stream of size " + ConfigInjectedManySymbols.testEventsNo)
      /*val testStream = generateStreamWithOrderSpecificInjectedDependencies(
        ConfigInjectedManySymbols.testEventsNo,
        ConfigInjectedManySymbols.maxOrder,
        (ConfigInjectedManySymbols.symbolsNo-1).toString,
        middleSymbols)*/
      val testStream = StreamGenerator.generateStreamWithDummyInjectedDependencies(
        ConfigInjectedManySymbols.testEventsNo,
        ConfigInjectedManySymbols.maxOrder,
        (ConfigInjectedManySymbols.symbolsNo - 1).toString,
        symbols.map(s => s.toString).toSet,
        ConfigInjectedManySymbols.target,
        ConfigInjectedManySymbols.symbolOrLtdProb,
        ConfigInjectedManySymbols.whichLtdProb
      )
      val psaTestStreamFn = ConfigInjectedManySymbols.resultsDir + "/streamTest" + test + ".csv"
      testStream.writeCSV(psaTestStreamFn)

      var row = List(test.toString, maxSpread, predictionThreshold)
      logger.info("Running forecasting with disambiguated SDFA")
      val sdfaProvDirect = SDFAProvider(SDFASourceDirect(List(sdfam)))
      val fsmDisProv = FSMProvider(sdfaProvDirect)
      val met = MatrixMLETask(fsmDisProv, ArrayStreamSource(trainStream))
      val mc = met.execute()._1
      val mcProv = MarkovChainProvider(MCSourceDirect(mc))
      val wtdProv = WtProvider(WtSourceMatrix(fsmDisProv, mcProv, ConfigInjectedManySymbols.horizon, ConfigInjectedManySymbols.finalsEnabled))
      val predProv = ForecasterProvider(ForecasterSourceBuild(fsmDisProv, wtdProv, ConfigInjectedManySymbols.horizon, predictionThreshold, maxSpread))
      val erfDisTask = ERFTask(fsmDisProv, predProv, ArrayStreamSource(testStream), show = true)
      val profilerDis = erfDisTask.execute()
      profilerDis.printProfileInfo()
      val (precDis, spreadDis, intervalScoreDis) = (profilerDis.getStatFor("precision", 0), profilerDis.getStatFor("spread", 0), profilerDis.getStatFor("interval", 0))
      row = row ::: List(sdfam.size.toString, precDis.toString, spreadDis.toString, intervalScoreDis.toString)

      for (m <- 1 to ConfigInjectedManySymbols.maxOrder) {
        logger.info("Learning SPSA")
        val maxNoStates = ConfigInjectedManySymbols.maxNoStates
        val (spsaLearnt, psaLearnt) = VMMUtils.learnSPSAFromSingleSDFA(sdfa0, m, ArrayStreamSource(trainStream), iso, maxNoStates, ConfigUtils.singlePartitionVal)
        logger.info("PSA learnt\n " + psaLearnt.toString)
        logger.info("PSA size/maxOrder/ordersCounts: " + psaLearnt.size + "/" + psaLearnt.maxOrder + "/" + psaLearnt.statesByOrder)
        val spsaLearntFn = ConfigInjectedManySymbols.resultsDir + "/learntPSATest" + test + " .spsa"
        val spsaLearnti = SPSAInterface(spsaLearnt, 1)
        //SerializationUtils.write2File[SPSAInterface](List(spsaLearnti),spsaLearntFn)

        logger.info("Running forecasting with learnt")
        val profilerLearnt = runFore(spsaLearnt, testStream, ConfigInjectedManySymbols.horizon, predictionThreshold, maxSpread)
        val (precLearnt, spreadLearnt, intervalScoreLearnt) = (profilerLearnt.getStatFor("precision", 0), profilerLearnt.getStatFor("spread", 0), profilerLearnt.getStatFor("interval", 0))
        row = row ::: List(m.toString, spsaLearnt.getSize.toString, precLearnt.toString, spreadLearnt.toString, intervalScoreLearnt.toString)
        logger.info("LEARNT@" + m + "\n Precision/Spread/Interval + \n" + precLearnt + "\n" + spreadLearnt + "\n" + intervalScoreLearnt)
      }
      logger.info("DIS\n Precision/Spread/Interval + \n" + precDis + "\n" + spreadDis + "\n" + intervalScoreDis)
      writer.writeRow(row)
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
    val wtdProv = WtProvider(WtSourceMatrix(fsmProv, matrixProv, horizon, ConfigInjectedManySymbols.finalsEnabled))
    val predProv = ForecasterProvider(ForecasterSourceBuild(fsmProv, wtdProv, horizon, predictionThreshold, maxSpread))
    val erf = ERFTask(fsmProv, predProv, ArrayStreamSource(stream), show = true)
    val profiler = erf.execute()
    profiler.printProfileInfo()
    profiler
  }

}
