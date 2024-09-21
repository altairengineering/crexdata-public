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
import workflow.provider.source.sdfa.{SDFASourceDFA, SDFASourceDirect}
import workflow.provider.source.spsa.SPSASourceDirect
import workflow.task.engineTask.LogLossTask
import workflow.task.estimatorTask.MatrixMLETask

object RunComparisonLogLoss extends LazyLogging {
  def main(args: Array[String]): Unit = {
    logger.info("Running ComparisonLogLoss with configuration " + ConfigLogLoss.toString)
    logger.info("Creating isomorphism")
    val symbols = (0 until ConfigLogLoss.symbolsNo).map(i => Symbol(i)).toList
    val simpleSentences = symbols.tail.map(s => AtomicSentence(PredicateConstructor.getEventTypePred(s.value.toString)).asInstanceOf[Sentence])
    val negatedSentences = simpleSentences.map(s => ComplexSentence(BooleanOperator.NOT, List(s)).asInstanceOf[Sentence]).toSet
    val negationsSentence = LogicUtils.connectWithAND(negatedSentences)
    val sentences = negationsSentence :: simpleSentences
    val iso = Isomorphism(sentences, symbols)

    val pattern = SymbolNode(ConfigLogLoss.target)

    logger.info("Building SDFA@0")
    logger.info("Building DFA")
    val dfaProvider0 = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, 0, symbols.map(s => s.toString).toSet))
    val dfa0 = dfaProvider0.provide().head.dfa
    logger.info("Using isomorphism to build SDFA")
    val sdfaProvider0 = SDFAProvider(SDFASourceDFA(dfa0, iso))
    val sdfa0 = sdfaProvider0.provide().head.sdfa

    logger.info("Building SDFA@" + ConfigLogLoss.disOrder)
    logger.info("Building DFA")
    val dfaProviderm = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, ConfigLogLoss.disOrder, symbols.map(s => s.toString).toSet))
    val dfam = dfaProviderm.provide().head.dfa
    logger.info("Using isomorphism to build SDFA")
    val sdfaProviderm = SDFAProvider(SDFASourceDFA(dfam, iso))
    val sdfam = sdfaProviderm.provide().head.sdfa

    for (test <- 1 to ConfigLogLoss.testsNo) {
      logger.info("Creating train stream of size " + ConfigLogLoss.trainEventsNo)
      val (psa, trainStream, testStream) = StreamGenerator.generatePSA(
        test,
        ConfigLogLoss.trainEventsNo,
        ConfigLogLoss.testEventsNo,
        (0 until ConfigLogLoss.symbolsNo).map(i => Symbol(i)).toList,
        ConfigLogLoss.spsaOrder,
        ConfigLogLoss.expansionProb,
        ConfigLogLoss.resultsDir
      )
      /*logger.info("Creating train stream of size " + ConfigLogLoss.trainEventsNo)
      val trainStream = StreamGenerator.generateStreamWithDummyInjectedDependencies(
        ConfigLogLoss.trainEventsNo,
        ConfigLogLoss.spsaOrder,
        (ConfigLogLoss.symbolsNo - 1).toString,
        symbols.map(s => s.toString).toSet,
        ConfigLogLoss.target,
        ConfigLogLoss.symbolOrLtdProb,
        ConfigLogLoss.whichLtdProb
      )
      logger.info("Creating test stream of size " + ConfigLogLoss.testEventsNo)
      val testStream = StreamGenerator.generateStreamWithDummyInjectedDependencies(
        ConfigLogLoss.testEventsNo,
        ConfigLogLoss.spsaOrder,
        (ConfigLogLoss.symbolsNo - 1).toString,
        symbols.map(s => s.toString).toSet,
        ConfigLogLoss.target,
        ConfigLogLoss.symbolOrLtdProb,
        ConfigLogLoss.whichLtdProb
      )*/

      var row = List(test.toString)

      logger.info("Running log-loss calculation with disambiguated SDFA")
      val sdfaProvDirect = SDFAProvider(SDFASourceDirect(List(sdfam)))
      val fsmDisProv = FSMProvider(sdfaProvDirect)
      val met = MatrixMLETask(fsmDisProv, ArrayStreamSource(trainStream))
      val mc = met.execute()._1
      val mcProv = MarkovChainProvider(MCSourceDirect(mc))
      val llTaskDis = LogLossTask(fsmDisProv, mcProv, testStream)
      val profilerDis = llTaskDis.execute()

      logger.info("Learning SPSA")
      val maxNoStates = ConfigLogLoss.maxNoStates
      val spsa = VMMUtils.learnSPSAFromSingleSDFA(sdfa0, ConfigLogLoss.spsaOrder, ArrayStreamSource(trainStream), iso, maxNoStates, ConfigUtils.singlePartitionVal) //psa.size)
      logger.info("PSA learnt\n ")
      logger.info("Running log-loss with learnt")
      val spsaProv = SPSAProvider(SPSASourceDirect(List(spsa._1)))
      val fsmProv = FSMProvider(spsaProv)
      val matrixProv = MarkovChainProvider(MCSourceSPSA(fsmProv))
      val llTaskLearnt = LogLossTask(fsmProv, matrixProv, testStream)
      val profilerLearnt = llTaskLearnt.execute()

      logger.info("Running log-loss calculation with merged SPSA")
      logger.info("Merging PSA")
      val spsaMerged = VMMUtils.embedPSAinSDFA(sdfa0, psa, iso)
      val spsaMergedp = SPSAProvider(SPSASourceDirect(List(spsaMerged)))
      val fsmProvSPSA = FSMProvider(spsaMergedp)
      val matrixProvSPSA = MarkovChainProvider(MCSourceSPSA(fsmProvSPSA))
      val llTaskMerged = LogLossTask(fsmProvSPSA, matrixProvSPSA, testStream)
      val profilerMerged = llTaskMerged.execute()

      logger.info("SDFA STATS")
      profilerDis.printProfileInfo()
      logger.info("SPSA (Learnt) STATS")
      profilerLearnt.printProfileInfo()
      logger.info("SPSA (Merged) STATS")
      profilerMerged.printProfileInfo()
    }
  }

}
