package scripts.sessionsNtests.spsa.synthetic

import com.github.tototoshi.csv.CSVWriter
import com.typesafe.scalalogging.LazyLogging
import fsm.SPSAInterface
import fsm.symbolic.sfa.logic.{AtomicSentence, LogicUtils, PredicateConstructor, Sentence}
import fsm.classical.pattern.regexp.SymbolNode
import profiler.WtProfiler
import stream.source.ArrayStreamSource
import stream.StreamFactory
import stream.array.EventStream
import model.vmm.pst.psa.PSAUtils
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

object RunComparisonIso extends LazyLogging {
  def main(args: Array[String]): Unit = {
    val home = System.getenv("WAYEB_HOME")
    val resultsDir = home + "/results/psa"
    val resultsFile = resultsDir + "/results.csv"
    val psaNo = 10
    val maxOrder = 2 //4
    val disMaxOrder = List(1) //List(1,2)
    val symbolsNo = 50 //7
    val symbols = (0 until symbolsNo).map(i => Symbol(i)).toList
    val expansionProb = 0.3 //0.7
    val trainEventsNo = 1000000
    val testEventsNo = 10000
    val testsNo = 1

    val horizon = 100
    val maxSpread = 10
    val predictionThreshold = 0.3

    val simpleSentences = symbols.tail.map(s => AtomicSentence(PredicateConstructor.getEventTypePred(s.value.toString)).asInstanceOf[Sentence])
    val negationsSentence = LogicUtils.connectWithAND(simpleSentences.toSet)
    val sentences = negationsSentence :: simpleSentences
    val iso = Isomorphism(sentences, symbols)

    val writer = CSVWriter.open(resultsFile, append = true)

    for (psaID <- 1 to psaNo) {
      logger.info("Running comparison for PSA " + psaID)

      logger.info("Creating PSA with ID " + psaID)
      var psa = PSAUtils.createPSA(symbols.toSet, maxOrder, expansionProb)
      while (psa.maxOrder != maxOrder) psa = PSAUtils.createPSA(symbols.toSet, maxOrder, expansionProb)
      val psaFn = resultsDir + "/psaOrder" + maxOrder + "Symbols" + symbolsNo + "ID" + psaID + ".psa"
      //SerializationUtils.write2File[ProbSuffixAutomaton](List(psa),psaFn)
      logger.info("PSA generated\n " + psa.toString)
      logger.info("PSA size/maxOrder/ordersCounts: " + psa.size + "/" + psa.maxOrder + "/" + psa.statesByOrder)

      logger.info("Creating train stream of size " + trainEventsNo)
      val psaTrainStreamFn = resultsDir + "/streamTrain" + psaID + ".csv"
      val (trainStream, _) = StreamFactory.getStream(psa, trainEventsNo)
      trainStream.writeCSV(psaTrainStreamFn)
      val eventCounters = trainStream.getCounters.toList
      //val eventTypes = eventCounters.map(c => c._1)
      val counters = eventCounters.map(c => c._2)
      val maxCounter = counters.max
      val maxEvent = eventCounters.filter(c => c._2 == maxCounter).head._1

      val pattern = SymbolNode(maxEvent)
      //val formula1 = CEPLUtils.re2cepl(pattern1)
      val dfaProvider = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, 0, symbols.map(s => s.toString).toSet))
      val dfa = dfaProvider.provide().head.dfa
      val sdfaProvider = SDFAProvider(SDFASourceDFA(dfa, iso))
      val sdfa = sdfaProvider.provide().head.sdfa

      logger.info("Now running for pattern with symbol " + maxEvent + " having " + maxCounter + " instances in training stream")
      logger.info("Merging PSA " + psaID)
      val spsaMerged = VMMUtils.embedPSAinSDFA(sdfa, psa, iso)
      val spsaMergedFn = resultsDir + "/mergedPSA" + psaID + " .spsa"
      val spsaMergedp = SPSAProvider(SPSASourceDirect(List(spsaMerged)))
      val spsaMergedi = spsaMergedp.provide()
      //SerializationUtils.write2File[SPSAInterface](spsaMergedi,spsaMergedFn)

      logger.info("Learning SPSA")
      val maxNoStates = psa.size // 1000
      val (spsaLearnt, psaLearnt) = VMMUtils.learnSPSAFromSingleSDFA(sdfa, maxOrder, ArrayStreamSource(trainStream), iso, maxNoStates, ConfigUtils.singlePartitionVal)
      logger.info("PSA learnt\n " + psaLearnt.toString)
      logger.info("PSA size/maxOrder/ordersCounts: " + psaLearnt.size + "/" + psaLearnt.maxOrder + "/" + psaLearnt.statesByOrder)
      val spsaLearntFn = resultsDir + "/learntPSA" + psaID + " .spsa"
      val spsaLearnti = SPSAInterface(spsaLearnt, 1)
      //SerializationUtils.write2File[SPSAInterface](List(spsaLearnti),spsaLearntFn)

      logger.info("Creating test stream of size " + testEventsNo)
      val psaTestStreamFn = resultsDir + "/streamTestPSA" + psaID + "test" + 1 + ".csv"
      val (testStream, _) = StreamFactory.getStream(psa, testEventsNo)
      testStream.writeCSV(psaTestStreamFn)

      logger.info("Running forecasting with merged")
      val profilerMerged = runFore(spsaMerged, testStream)

      logger.info("Running forecasting with learnt")
      val profilerLearnt = runFore(spsaLearnt, testStream)

      val (precMerged, spreadMerged, intervalScoreMerged) = (profilerMerged.getStatFor("precision", 0), profilerMerged.getStatFor("spread", 0), profilerMerged.getStatFor("interval", 0))
      val (precLearnt, spreadLearnt, intervalScoreLearnt) = (profilerLearnt.getStatFor("precision", 0), profilerLearnt.getStatFor("spread", 0), profilerLearnt.getStatFor("interval", 0))

      var row = List(psaID.toString, spsaMerged.getSize.toString, precMerged.toString, spreadMerged.toString, intervalScoreMerged.toString,
                     spsaLearnt.getSize.toString, precLearnt.toString, spreadLearnt.toString, intervalScoreLearnt.toString)

      for (dm <- disMaxOrder) {
        logger.info("Creating disambiguated SDFA")
        //val disOrder = getDisOrder
        val dfaProviderM = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, dm, symbols.map(s => s.toString).toSet))
        val dfaM = dfaProviderM.provide().head.dfa
        val sdfaDisProvM = SDFAProvider(SDFASourceDFA(dfaM, iso))
        val sdfaDisM = sdfaDisProvM.provide().head.sdfa
        val sdfaProvDirect = SDFAProvider(SDFASourceDirect(List(sdfaDisM)))
        val fsmDisProv = FSMProvider(sdfaProvDirect)
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
        logger.info("DIS\n Precision/Spread/Interval + \n" + precDis + "\n" + spreadDis + "\n" + intervalScoreDis)

        row = row ::: List(dm.toString, sdfaDisM.size.toString, precDis.toString, spreadDis.toString, intervalScoreDis.toString)

      }
      writer.writeRow(row)
      logger.info("MERGED\n Precision/Spread/Interval + \n" + precMerged + "\n" + spreadMerged + "\n" + intervalScoreMerged)
      logger.info("LEARNT\n Precision/Spread/Interval + \n" + precLearnt + "\n" + spreadLearnt + "\n" + intervalScoreLearnt)
    }

      def runFore(
          spsa: SymbolicPSA,
          stream: EventStream
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

  }

}
