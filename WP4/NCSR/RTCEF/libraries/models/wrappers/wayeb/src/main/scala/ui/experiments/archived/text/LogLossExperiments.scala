package ui.experiments.archived.text

import com.typesafe.scalalogging.LazyLogging
import fsm.symbolic.sfa.logic.{AtomicSentence, ComplexSentence, LogicUtils, PredicateConstructor, Sentence}
import fsm.symbolic.sre.BooleanOperator
import fsm.classical.pattern.regexp.SymbolNode
import stream.StreamFactory
import stream.source.{ArrayStreamSource, EmitMode, StreamSource}
import model.vmm.{Isomorphism, Symbol, VMMUtils}
import ui.ConfigUtils
import workflow.provider.{DFAProvider, FSMProvider, MarkovChainProvider, SDFAProvider, SPSAProvider}
import workflow.provider.source.dfa.DFASourceRegExp
import workflow.provider.source.matrix.{MCSourceDirect, MCSourceSPSA}
import workflow.provider.source.sdfa.{SDFASourceDFA, SDFASourceDirect}
import workflow.provider.source.spsa.SPSASourceDirect
import workflow.task.engineTask.LogLossTask
import workflow.task.estimatorTask.{HMMTask, MatrixMLETask}

object LogLossExperiments extends LazyLogging {
  def main(args: Array[String]): Unit = {
    val alphabet: List[String] = List("SPACE", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
    val symbols = (0 until alphabet.length).map(i => Symbol(i)).toList
    val simpleSentences = alphabet.tail.map(s => AtomicSentence(PredicateConstructor.getEventTypePred(s)).asInstanceOf[Sentence])
    val negatedSentences = simpleSentences.map(s => ComplexSentence(BooleanOperator.NOT, List(s)).asInstanceOf[Sentence]).toSet
    val negationsSentence = LogicUtils.connectWithAND(negatedSentences)
    val sentences = negationsSentence :: simpleSentences
    val iso = Isomorphism(sentences, symbols)

    val pattern = SymbolNode("0")

    logger.info("Building SDFA@0")
    logger.info("Building DFA")
    val dfaProvider0 = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, 0, symbols.map(s => s.toString).toSet))
    val dfa0 = dfaProvider0.provide().head.dfa
    logger.info("Using isomorphism to build SDFA")
    val sdfaProvider0 = SDFAProvider(SDFASourceDFA(dfa0, iso))
    val sdfa0 = sdfaProvider0.provide().head.sdfa
    logger.info("SDFA@0: " + sdfa0.toString)

    val disOrder = 1
    logger.info("Building SDFA@" + disOrder)
    logger.info("Building DFA")
    val dfaProviderm = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, disOrder, symbols.map(s => s.toString).toSet))
    val dfam = dfaProviderm.provide().head.dfa
    logger.info("Using isomorphism to build SDFA")
    val sdfaProviderm = SDFAProvider(SDFASourceDFA(dfam, iso))
    val sdfam = sdfaProviderm.provide().head.sdfa
    logger.info("SDFA@m has " + sdfam.size + " states")

    logger.info("Creating stream sources for folds")

    val folds = List(1) //,2,3,4)
    val domain = "text"
    val foldsDir = "/home/zmithereen/data/text/stream/folds"
    val streamSources: List[(StreamSource, StreamSource)] = for (f <- folds) yield (
      StreamFactory.getDomainStreamSource(foldsDir + "/fold" + f + "_train.csv", domain, List.empty),
      StreamFactory.getDomainStreamSource(foldsDir + "/fold" + f + "_test.csv", domain, List.empty)
    )

    for (fold <- folds) {
      val trainStreamSource = streamSources(fold - 1)._1
      val trainStream = trainStreamSource.emitEventsAndClose(EmitMode.BUFFER)
      val testStreamSource = streamSources(fold - 1)._2
      val testStream = testStreamSource.emitEventsAndClose(EmitMode.BUFFER)

      logger.info("Running log-loss calculation with disambiguated SDFA")
      val sdfaProvDirect = SDFAProvider(SDFASourceDirect(List(sdfam)))
      val fsmDisProv = FSMProvider(sdfaProvDirect)
      val met = MatrixMLETask(fsmDisProv, ArrayStreamSource(trainStream))
      val mc = met.execute()._1
      val mcProv = MarkovChainProvider(MCSourceDirect(mc))
      val llTaskDis = LogLossTask(fsmDisProv, mcProv, testStream)
      val profilerDis = llTaskDis.execute()
      profilerDis.printProfileInfo()

      logger.info("Running HMM")
      val hmmt = HMMTask(fsmDisProv, trainStreamSource)
      val hmm = hmmt.execute()._1

      /*logger.info("Learning SPSA")
      val maxNoStates = 10000
      val spsaOrder = 30
      val spsa = VMMUtils.learnSPSAFromSingleSDFA(sdfa0,spsaOrder,ArrayStreamSource(trainStream),iso,maxNoStates, ConfigUtils.singlePartitionVal)//psa.size)
      logger.info("PSA learnt\n ")
      logger.info("Running log-loss with learnt")
      val spsaProv = SPSAProvider(SPSASourceDirect(List(spsa._1)))
      val fsmProv = FSMProvider(spsaProv)
      val matrixProv = MatrixProvider(MatrixSourceSPSA(fsmProv))
      val llTaskLearnt = LogLossTask(fsmProv,matrixProv,testStream)
      val profilerLearnt = llTaskLearnt.execute()
      profilerLearnt.printProfileInfo()*/
    }
  }

}
