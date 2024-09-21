package scripts.sessionsNtests.spsa.synthetic

import com.github.tototoshi.csv.CSVWriter
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
import workflow.provider.source.wt.{WtSourceDirect, WtSourceMatrix}
import workflow.task.engineTask.archived.FixedDistanceTask
import workflow.task.estimatorTask.MatrixMLETask

object RunComparisonFixedDistance extends LazyLogging {
  def main(args: Array[String]): Unit = {
    logger.info("Running ComparisonFixedDistance with configuration " + ConfigFixedDistance.toString)
    logger.info("Creating isomorphism")
    val symbols = (0 until ConfigFixedDistance.symbolsNo).map(i => Symbol(i)).toList
    val simpleSentences = symbols.tail.map(s => AtomicSentence(PredicateConstructor.getEventTypePred(s.value.toString)).asInstanceOf[Sentence])
    val negatedSentences = simpleSentences.map(s => ComplexSentence(BooleanOperator.NOT, List(s)).asInstanceOf[Sentence]).toSet
    val negationsSentence = LogicUtils.connectWithAND(negatedSentences)
    val sentences = negationsSentence :: simpleSentences
    val iso = Isomorphism(sentences, symbols)

    val pattern = SymbolNode(ConfigFixedDistance.target)

    logger.info("Building SDFA@0")
    logger.info("Building DFA")
    val dfaProvider0 = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, 0, symbols.map(s => s.toString).toSet))
    val dfa0 = dfaProvider0.provide().head.dfa
    logger.info("Using isomorphism to build SDFA")
    val sdfaProvider0 = SDFAProvider(SDFASourceDFA(dfa0, iso))
    val sdfa0 = sdfaProvider0.provide().head.sdfa

    logger.info("Building SDFA@" + ConfigFixedDistance.disOrder)
    logger.info("Building DFA")
    val dfaProviderm = DFAProvider(DFASourceRegExp(pattern, ConfigUtils.defaultPolicy, ConfigFixedDistance.disOrder, symbols.map(s => s.toString).toSet))
    val dfam = dfaProviderm.provide().head.dfa
    logger.info("Using isomorphism to build SDFA")
    val sdfaProviderm = SDFAProvider(SDFASourceDFA(dfam, iso))
    val sdfam = sdfaProviderm.provide().head.sdfa

    val writer = CSVWriter.open(ConfigFixedDistance.resultsFile, append = true)
    for (test <- 1 to ConfigFixedDistance.testsNo) {
      /*logger.info("Creating train stream of size " + ConfigFixedDistance.trainEventsNo)
      val (psa,trainStream,testStream) = StreamGenerator.generatePSA(
        test,
        ConfigFixedDistance.trainEventsNo,
        ConfigFixedDistance.testEventsNo,
        (0 until ConfigFixedDistance.symbolsNo).map(i => Symbol(i)).toList,
        ConfigFixedDistance.spsaOrder,
        ConfigFixedDistance.expansionProb,
        ConfigFixedDistance.resultsDir)*/
      val defaultSymbol = (ConfigFixedDistance.symbolsNo - 1).toString
      val symbolsNotDefault = symbols.map(s => s.toString).toSet - defaultSymbol
      val ltdg = new LtdGenerator2(
        ConfigFixedDistance.target,
        ConfigFixedDistance.spsaOrder,
        symbolsNotDefault,
        ConfigFixedDistance.whichLtdProb
      )
      logger.info("Creating train stream of size " + ConfigFixedDistance.trainEventsNo)
      val trainStream = StreamGenerator.generateStreamWithSpecificLtd(
        ConfigFixedDistance.trainEventsNo,
        ConfigFixedDistance.spsaOrder,
        defaultSymbol,
        ConfigFixedDistance.symbolOrLtdProb,
        ltdg
      )
      val psaTrainStreamFn = ConfigFixedDistance.resultsDir + "/streamTrain" + test + ".csv"
      //trainStream.writeCSV(psaTrainStreamFn)
      logger.info("Creating test stream of size " + ConfigFixedDistance.testEventsNo)
      val untaggedTestStream = StreamGenerator.generateStreamWithSpecificLtd(
        ConfigFixedDistance.testEventsNo,
        ConfigFixedDistance.spsaOrder,
        defaultSymbol,
        ConfigFixedDistance.symbolOrLtdProb,
        ltdg
      )
      val psaTestStreamFn = ConfigFixedDistance.resultsDir + "/streamTest" + test + ".csv"
      //testStream.writeCSV(psaTestStreamFn)

      logger.info("Learning SDFA Matrix")
      val sdfaProvDirect = SDFAProvider(SDFASourceDirect(List(sdfam)))
      val fsmDisProv = FSMProvider(sdfaProvDirect)
      val met = MatrixMLETask(fsmDisProv, ArrayStreamSource(trainStream))
      val mc = met.execute()._1
      val mcProv = MarkovChainProvider(MCSourceDirect(mc))
      val wtdProvDis = WtProvider(WtSourceMatrix(fsmDisProv, mcProv, ConfigFixedDistance.horizon, ConfigFixedDistance.finalsEnabled))
      val wtdsDis = wtdProvDis.provide()
      logger.info("SDFA Matrix learnt")

      logger.info("Learning SPSA")
      val maxNoStates = ConfigFixedDistance.maxNoStates
      val spsa = VMMUtils.learnSPSAFromSingleSDFA(sdfa0, ConfigFixedDistance.spsaOrder, ArrayStreamSource(trainStream), iso, maxNoStates, ConfigUtils.singlePartitionVal) //psa.size)
      val spsaProv = SPSAProvider(SPSASourceDirect(List(spsa._1)))
      val fsmProv = FSMProvider(spsaProv)
      val matrixProv = MarkovChainProvider(MCSourceSPSA(fsmProv))
      val wtdProvLearnt = WtProvider(WtSourceMatrix(fsmProv, matrixProv, ConfigFixedDistance.horizon, ConfigFixedDistance.finalsEnabled))
      val wtdsLearnt = wtdProvLearnt.provide()
      logger.info("PSA learnt\n ")
      //logger.info(spsa._2.toString)
      logger.info(spsa._1.toString)
      //val spsaLearntFn = ConfigFixedDistance.resultsDir + "/learntPSA" + test + " .spsa"
      //val spsaLearnti = SPSAInterface(spsa._1,1)
      //SerializationUtils.write2File[SPSAInterface](List(spsaLearnti),spsaLearntFn)

      /*logger.info("Merging PSA")
      val spsaMerged = VMMUtils.mergeSingleSFAPSA(sdfa0, psa, iso)
      val spsaMergedp = SPSAProvider(new SPSASourceDirect(List(spsaMerged)))
      val fsmProvSPSA = FSMProvider(spsaMergedp)
      val matrixProvSPSA = MatrixProvider(MatrixSourceSPSA(fsmProvSPSA))
      logger.info("Merging done")*/

      for (maxSpread <- ConfigFixedDistance.maxSpread; distance <- ConfigFixedDistance.distance; predictionThreshold <- ConfigFixedDistance.predictionThreshold) {
        var row = List(test.toString, distance.toString, maxSpread.toString, predictionThreshold.toString)

        /*logger.info("Creating test stream of size " + ConfigFixedDistance.testEventsNo)
        val testStream = StreamGenerator.generateTaggedStreamWithSpecificLtd(
          ConfigFixedDistance.testEventsNo,
          ConfigFixedDistance.spsaOrder,
          defaultSymbol,
          ConfigFixedDistance.symbolOrLtdProb,
          ltdg,
          distance
        )
        val psaTestStreamFn = ConfigFixedDistance.resultsDir + "/streamTest" + test + ".csv"
        testStream.writeCSV(psaTestStreamFn)*/
        val testStream = StreamGenerator.tagStreamAtDistance(untaggedTestStream, distance)

        logger.info("Running fixed distance with disambiguated SDFA")
        val pp = ForecasterProvider(ForecasterSourceBuild(
          fsmDisProv,
          WtProvider(WtSourceDirect(wtdsDis)),
          ConfigFixedDistance.horizon,
          predictionThreshold,
          maxSpread,
          ConfigFixedDistance.spreadMethod
        ))
        //val fdTaskDis = FixedDistanceTask(fsmDisProv, pp, (distance,distance), testStream)
        val fdTaskDis = FixedDistanceTask(fsmDisProv, pp, ConfigFixedDistance.checkForEmitting, ConfigFixedDistance.finalsEnabled, testStream)
        val profilerDis = fdTaskDis.execute()

        logger.info("Running fixed distance with learnt")
        val ppLearnt = ForecasterProvider(ForecasterSourceBuild(
          fsmProv,
          WtProvider(WtSourceDirect(wtdsLearnt)),
          ConfigFixedDistance.horizon,
          predictionThreshold,
          maxSpread,
          ConfigFixedDistance.spreadMethod
        ))
        //val fdTaskLearnt = FixedDistanceTask(fsmProv, ppLearnt, (distance,distance), testStream)
        val fdTaskLearnt = FixedDistanceTask(fsmProv, ppLearnt, ConfigFixedDistance.checkForEmitting, ConfigFixedDistance.finalsEnabled, testStream)
        val profilerLearnt = fdTaskLearnt.execute()

        /*logger.info("Running fixed distance with merged SPSA")
        val ppMerged = PredictorProvider(PredictorSourceBuild(
          fsmProvSPSA,
          matrixProvSPSA,
          ConfigFixedDistance.horizon,
          predictionThreshold,
          maxSpread,
          ConfigFixedDistance.spreadMethod))
        val fdTaskMerged = FixedDistanceTask(fsmProvSPSA, ppMerged, (distance,distance), testStream)
        val profilerMerged = fdTaskMerged.execute()*/

        logger.info("SDFA STATS")
        profilerDis.printProfileInfo()
        logger.info("SPSA (Learnt) STATS")
        profilerLearnt.printProfileInfo()
        /*logger.info("SPSA (Merged) STATS")
        profilerMerged.printProfileInfo()*/

        val (precDis, spreadDis, intervalScoreDis, predictionsNoDis, lossRatioDis, distDis, confDis, devDis) = (
          profilerDis.getStatFor("precision", 0),
          profilerDis.getStatFor("spread", 0),
          profilerDis.getStatFor("interval", 0),
          profilerDis.getStatFor("predictionsNo", 0),
          profilerDis.getStatFor("lossRatio", 0),
          profilerDis.getStatFor("distance", 0),
          profilerDis.getStatFor("confidence", 0),
          profilerDis.getStatFor("deviation", 0)
        )
        //val inverseSpreadDis = (ConfigFixedDistance.horizon - spreadDis.toDouble)/ConfigFixedDistance.horizon
        //val negIntervalScoreDis = -intervalScoreDis.toDouble
        val recallDis = 1.0 - lossRatioDis.toDouble
        row = row ::: List(
          sdfam.size.toString,
          precDis.toString,
          spreadDis.toString,
          intervalScoreDis.toString,
          recallDis.toString,
          distDis.toString,
          confDis.toString,
          devDis.toString,
          predictionsNoDis.toString
        )
        /*val (precMerged, spreadMerged, intervalScoreMerged, predictionsNoMerged, lossRatioMerged) = (
          profilerMerged.getStatFor("precision", 0),
          profilerMerged.getStatFor("spread", 0),
          profilerMerged.getStatFor("interval", 0),
          profilerMerged.getStatFor("predictionsNo", 0),
          profilerMerged.getStatFor("lossRatio", 0))
        row = row:::List(spsaMerged.getSize.toString,precMerged.toString,spreadMerged.toString,intervalScoreMerged.toString,lossRatioMerged.toString,predictionsNoMerged.toString)*/
        val (precLearnt, spreadLearnt, intervalScoreLearnt, predictionsNoLearnt, lossRatioLearnt, distLearnt, confLearnt, devLearnt) = (
          profilerLearnt.getStatFor("precision", 0),
          profilerLearnt.getStatFor("spread", 0),
          profilerLearnt.getStatFor("interval", 0),
          profilerLearnt.getStatFor("predictionsNo", 0),
          profilerLearnt.getStatFor("lossRatio", 0),
          profilerLearnt.getStatFor("distance", 0),
          profilerLearnt.getStatFor("confidence", 0),
          profilerLearnt.getStatFor("deviation", 0)
        )
        //val inverseSpreadLearnt = (ConfigFixedDistance.horizon - spreadLearnt.toDouble)/ConfigFixedDistance.horizon
        //val negIntervalScoreLearnt = -intervalScoreLearnt.toDouble
        val recallLearnt = 1.0 - lossRatioLearnt.toDouble
        row = row ::: List(
          spsa._1.getSize.toString,
          precLearnt.toString,
          spreadLearnt.toString,
          intervalScoreLearnt.toString,
          recallLearnt.toString,
          distLearnt.toString,
          confLearnt.toString,
          devLearnt.toString,
          predictionsNoLearnt.toString
        )

        //logger.info("MERGED\n Precision/Spread/Interval + \n" + precMerged + "\n" + spreadMerged + "\n" + intervalScoreMerged)
        logger.info("DIS\n Precision/Spread/Interval + \n" + precDis + "\n" + spreadDis + "\n" + intervalScoreDis)
        logger.info("LEARNT\n Precision/Spread/Interval + \n" + precLearnt + "\n" + spreadLearnt + "\n" + intervalScoreLearnt)

        writer.writeRow(row)
      }
    }
    writer.close()
  }

}
