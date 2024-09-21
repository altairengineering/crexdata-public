package scripts.sessionsNtests.spsa.synthetic

import com.typesafe.scalalogging.LazyLogging
import fsm.symbolic.sre.BooleanOperator
import fsm.symbolic.sfa.logic._
import fsm.classical.pattern.regexp.SymbolNode
import stream.source.ArrayStreamSource
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
import workflow.task.engineTask.ERFNTask
import workflow.task.estimatorTask.MatrixMLETask

object RunComparisonNext extends LazyLogging {
  def main(args: Array[String]): Unit = {
    logger.info("Running ComparisonInjectedManySymbols with configuration " + ConfigNext.toString)
    logger.info("Creating isomorphism")
    val symbols = (0 until ConfigNext.symbolsNo).map(i => Symbol(i)).toList
    val simpleSentences = symbols.tail.map(s => AtomicSentence(PredicateConstructor.getEventTypePred(s.value.toString)).asInstanceOf[Sentence])
    val negatedSentences = simpleSentences.map(s => ComplexSentence(BooleanOperator.NOT, List(s)).asInstanceOf[Sentence]).toSet
    val negationsSentence = LogicUtils.connectWithAND(negatedSentences)
    val sentences = negationsSentence :: simpleSentences
    val iso = Isomorphism(sentences, symbols)

    val patterns = symbols.map(s => SymbolNode(s.toString))

    logger.info("Building SDFA@0")
    logger.info("Building DFA")
    val dfaProvider0 = patterns.map(pattern => DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, 0, symbols.map(s => s.toString).toSet)))
    val dfa0 = dfaProvider0.map(p => p.provide().head.dfa)
    logger.info("Using isomorphism to build SDFA")
    val sdfaProvider0 = dfa0.map(d => SDFAProvider(SDFASourceDFA(d, iso)))
    val sdfa0 = sdfaProvider0.map(s => s.provide().head.sdfa)

    logger.info("Building SDFA@" + ConfigNext.disMaxOrder)
    logger.info("Building DFA")
    val dfaProviderm = patterns.map(pattern => DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, ConfigNext.disMaxOrder, symbols.map(s => s.toString).toSet)))
    val dfam = dfaProviderm.map(p => p.provide().head.dfa)
    logger.info("Using isomorphism to build SDFA")
    val sdfaProviderm = dfam.map(d => SDFAProvider(SDFASourceDFA(d, iso)))
    val sdfam = sdfaProviderm.map(s => s.provide().head.sdfa)

    for (test <- 1 to ConfigNext.testsNo) {
      /*logger.info("Creating train stream of size " + ConfigNext.trainEventsNo)
      val (psa,trainStream,testStream) = StreamGenerator.generatePSA(
        test,
        ConfigNext.trainEventsNo,
        ConfigNext.testEventsNo,
        (1 until ConfigNext.symbolsNo).map(i => Symbol(i)).toList,
        ConfigNext.maxOrder,
        ConfigNext.expansionProb,
        ConfigNext.resultsDir)*/
      logger.info("Creating train stream of size " + ConfigNext.trainEventsNo)
      val trainStream = StreamGenerator.generateStreamWithDummyInjectedDependencies(
        ConfigNext.trainEventsNo,
        ConfigNext.maxOrder,
        (ConfigNext.symbolsNo - 1).toString,
        symbols.map(s => s.toString).toSet,
        ConfigNext.target,
        ConfigNext.symbolOrLtdProb,
        ConfigNext.whichLtdProb
      )
      logger.info("Creating test stream of size " + ConfigNext.testEventsNo)
      val testStream = StreamGenerator.generateStreamWithDummyInjectedDependencies(
        ConfigNext.testEventsNo,
        ConfigNext.maxOrder,
        (ConfigNext.symbolsNo - 1).toString,
        symbols.map(s => s.toString).toSet,
        ConfigNext.target,
        ConfigNext.symbolOrLtdProb,
        ConfigNext.whichLtdProb
      )
      val psaTestStreamFn = ConfigNext.resultsDir + "/streamTest" + test + ".csv"
      testStream.writeCSV(psaTestStreamFn)

      var row = List(test.toString)
      logger.info("Running forecasting with disambiguated SDFA")
      val sdfaProvDirect = SDFAProvider(SDFASourceDirect(sdfam))
      val fsmDisProv = FSMProvider(sdfaProvDirect)
      val met = MatrixMLETask(fsmDisProv, ArrayStreamSource(trainStream))
      val mc = met.execute()._1
      val mcProv = MarkovChainProvider(MCSourceDirect(mc))
      val wtdProv = WtProvider(WtSourceMatrix(fsmDisProv, mcProv, ConfigNext.horizon, ConfigNext.finalsEnabled))
      val predProv = ForecasterProvider(ForecasterSourceBuild(fsmDisProv, wtdProv, ConfigNext.horizon, ConfigNext.predictionThreshold, ConfigNext.maxSpread))
      val erfDisTask = ERFNTask(fsmDisProv, predProv, 1, ArrayStreamSource(testStream))
      val profilerDis = erfDisTask.execute()

      //val (precDis, spreadDis, intervalScoreDis) = (profilerDis.getStatFor("precision",0), profilerDis.getStatFor("spread",0), profilerDis.getStatFor("interval",0))
      //row = row:::List(sdfam.size.toString,precDis.toString,spreadDis.toString,intervalScoreDis.toString)

      for (m <- 1 to ConfigNext.maxOrder) {
        logger.info("Learning SPSA")
        val maxNoStates = ConfigNext.maxNoStates
        val spsa = sdfa0.map(s => VMMUtils.learnSPSAFromSingleSDFA(s, m, ArrayStreamSource(trainStream), iso, maxNoStates, ConfigUtils.singlePartitionVal))
        logger.info("PSA learnt\n ")

        logger.info("Running forecasting with learnt")
        val spsaProv = SPSAProvider(SPSASourceDirect(spsa.map(s => s._1)))
        val fsmProv = FSMProvider(spsaProv)
        val matrixProv = MarkovChainProvider(MCSourceSPSA(fsmProv))
        val wtdProv = WtProvider(WtSourceMatrix(fsmProv, matrixProv, ConfigNext.horizon, ConfigNext.finalsEnabled))
        val predProv = ForecasterProvider(ForecasterSourceBuild(fsmProv, wtdProv, ConfigNext.horizon, ConfigNext.predictionThreshold, ConfigNext.maxSpread))
        val erf = ERFNTask(fsmProv, predProv, 1, ArrayStreamSource(testStream))
        val profiler = erf.execute()
        profiler.printProfileInfo()
      }
      profilerDis.printProfileInfo()
    }
  }

}
