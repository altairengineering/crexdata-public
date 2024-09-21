package scripts.sessionsNtests.spsa.synthetic

import com.github.tototoshi.csv.CSVWriter
import com.typesafe.scalalogging.LazyLogging
import fsm.SPSAInterface
import fsm.symbolic.sre.SREUtils
import fsm.symbolic.sfa.SFAUtils
import fsm.symbolic.sfa.logic.{AtomicSentence, PredicateConstructor, Sentence}
import fsm.symbolic.sfa.sdfa.SDFAUtils
import fsm.symbolic.sfa.snfa.SNFAUtils
import fsm.classical.pattern.regexp.OperatorType.CONCAT
import fsm.classical.pattern.regexp.{OperatorNode, SymbolNode}
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

object RunComparison extends LazyLogging {
  def main(args: Array[String]): Unit = {
    val home = System.getenv("WAYEB_HOME")
    val resultsDir = home + "/results/psa"
    val resultsFile = resultsDir + "/results.csv"
    val psaAnalysisFile = resultsDir + "/psaAnalysis.csv"
    val minTermMethod = "withoutsat"
    val psaNo = 50
    val maxOrder = 10 //4
    val disMaxOrder = List(1, 5, 10) //List(1,2)
    val symbolsNo = 1 //7
    val symbols = (1 to symbolsNo).map(i => Symbol(i)).toList
    val expansionProb = 0.8 //0.7
    val trainEventsNo = 1000000
    val testEventsNo = 10000
    val testsNo = 1

    val horizon = 200
    val maxSpread = horizon
    val predictionThreshold = 0.5

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
    val pattern12 = OperatorNode(CONCAT, List(SymbolNode("1"), SymbolNode("2")))
    val pattern123 = OperatorNode(CONCAT, List(pattern12, SymbolNode("3")))

    val patterns = List(pattern1)
    val formulas = patterns.map(p => SREUtils.re2formula(p))

    val formulasArray = formulas.toArray

    val writer = CSVWriter.open(resultsFile, append = true)
    val psaAnalysisWriter = CSVWriter.open(psaAnalysisFile, append = true)

    for (psaID <- 1 to psaNo) {
      logger.info("Running comparison for PSA " + psaID)

      val (psa, trainStream, testStream) = StreamGenerator.generatePSA(psaID, trainEventsNo, testEventsNo, symbols, maxOrder, expansionProb, resultsDir)

      logger.info("Now running for patterns")
      for (f <- formulasArray.indices) {
        val formula = formulasArray(f)
        val snfaStream = SNFAUtils.buildSNFAForStream(formula)
        val sdfa = SFAUtils.determinizeI(snfaStream, exclusives, extras, minTermMethod)
        SDFAUtils.checkForDead(sdfa)
        val sdfaPol = SDFAUtils.setPolicy(sdfa, policy)
        val sentences = sdfaPol.getSentences
        val minTermsWithoutNeg = symbols.map(symbol => sentences.filter(sentence => sentence.evaluate(GenericEvent(symbol.value.toString, 1))).head)
        val negationSentence = sentences.filter(sentence => !symbols.exists(symbol => sentence.evaluate(GenericEvent(symbol.value.toString, 1)))).head
        val minTerms = negationSentence :: minTermsWithoutNeg
        val isoSymbols = Symbol(0) :: symbols
        val iso = Isomorphism(minTerms, isoSymbols)

        logger.info("Merging PSA " + psaID + " and SDFA " + f)
        val spsaMerged = VMMUtils.embedPSAinSDFA(sdfaPol, psa, iso)
        val spsaMergedFn = resultsDir + "/mergedPSA" + psaID + "Pattern" + f + " .spsa"
        val spsaMergedp = SPSAProvider(SPSASourceDirect(List(spsaMerged)))
        val spsaMergedi = spsaMergedp.provide()
        //SerializationUtils.write2File[SPSAInterface](spsaMergedi,spsaMergedFn)

        logger.info("Learning SPSA")
        val maxNoStates = psa.size // 1000
        val (spsaLearnt, part, psaLearnt) = VMMUtils.learnSPSAFromSingleFormula((formula, maxOrder, "$"), exclusives, extras, policy, ArrayStreamSource(trainStream), maxNoStates, iso)
        logger.info("PSA learnt\n " + psaLearnt.toString)
        logger.info("PSA size/maxOrder/ordersCounts: " + psaLearnt.size + "/" + psaLearnt.maxOrder + "/" + psaLearnt.statesByOrder)
        val spsaLearntFn = resultsDir + "/learntPSA" + psaID + "Pattern" + f + " .spsa"
        val spsaLearnti = SPSAInterface(spsaLearnt, 1)
        //SerializationUtils.write2File[SPSAInterface](List(spsaLearnti),spsaLearntFn)

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
          val formulasForDis = List((formula, dm, "$"))
          val sdfaDisProvF = SDFAProvider(new SDFASourceFormula(formulasForDis, policy, exclusives, extras, minTermMethod))
          val sdfaDis = sdfaDisProvF.provide()
          val sdfaDisProv = SDFAProvider(SDFASourceDirect(sdfaDis.map(el => el.sdfa)))
          val fsmDisProv = FSMProvider(sdfaDisProv)
          val met = MatrixMLETask(fsmDisProv, ArrayStreamSource(trainStream))
          val mc = met.execute()._1
          val mcProv = MarkovChainProvider(new MCSourceDirect(mc))
          val wtdProv = WtProvider(WtSourceMatrix(fsmDisProv, mcProv, horizon, false))
          val predProv = ForecasterProvider(ForecasterSourceBuild(fsmDisProv, wtdProv, horizon, predictionThreshold, maxSpread))

          logger.info("Running forecasting with disambiguated SDFA")
          val erfDisTask = ERFTask(fsmDisProv, predProv, ArrayStreamSource(testStream), show = true)
          val profilerDis = erfDisTask.execute()
          profilerDis.printProfileInfo()

          val (precDis, spreadDis, intervalScoreDis) = (profilerDis.getStatFor("precision", 0), profilerDis.getStatFor("spread", 0), profilerDis.getStatFor("interval", 0))
          logger.info("DIS\n Precision/Spread/Interval + \n" + precDis + "\n" + spreadDis + "\n" + intervalScoreDis)

          row = row ::: List(dm.toString, sdfaDis.head.size.toString, precDis.toString, spreadDis.toString, intervalScoreDis.toString)

        }
        writer.writeRow(row)
        logger.info("MERGED\n Precision/Spread/Interval + \n" + precMerged + "\n" + spreadMerged + "\n" + intervalScoreMerged)
        logger.info("LEARNT\n Precision/Spread/Interval + \n" + precLearnt + "\n" + spreadLearnt + "\n" + intervalScoreLearnt)

      }

    }

    writer.close()
    psaAnalysisWriter.close()

      def getDisOrder: Int = {
        val disMaxOrder = 5
        if (maxOrder >= disMaxOrder) disMaxOrder
        else maxOrder - 1
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
