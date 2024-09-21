package ui.experiments.exposed2cli

import com.typesafe.scalalogging.LazyLogging
import fsm.symbolic.sfa.SFAUtils
import fsm.symbolic.sfa.sdfa.SDFAUtils
import fsm.symbolic.sfa.snfa.SNFAUtils
import fsm.symbolic.sre.SREUtils
import model.waitingTime.ForecastMethod.ForecastMethod
import stream.StreamFactory
import workflow.provider._
import fsm.CountPolicy.CountPolicy
import stream.source.{EmitMode, StreamSource}
import workflow.provider.source.matrix.MCSourceDirect
import workflow.provider.source.forecaster.ForecasterSourceDirect
import workflow.provider.source.sdfa.{SDFASourceDirect, SDFASourceDirectI}
import workflow.provider.source.wt.WtSourceDirect
import workflow.task.engineTask.{ERFTask, LogLossTask}
import workflow.task.estimatorTask.MatrixMLETask
import workflow.task.predictorTask.WtPredictorTask
import model.waitingTime.ForecastMethod
import ui.ConfigUtils

object PatternExperimentsSDFA extends LazyLogging {

  def RunExperiments(
                      domain: String,
                      foldsDir: String,
                      folds: List[Int],
                      patternFilePath: String,
                      patternName: String,
                      declarationsFilePath: String,
                      resultsDir: String,
                      sdfaOrders: List[Int],
                      horizon: Int,
                      finalsEnabled: Boolean,
                      distances: List[(Double, Double)],
                      maxSpreads: List[Int],
                      thresholds: List[Double],
                      spreadMethod: ForecastMethod,
                      policy: CountPolicy,
                      maxSize: Int,
                      target: String
                    ): Unit = {

    logger.info("Creating stream sources for folds")

    val streamSources: List[(StreamSource, StreamSource)] = for (f <- folds) yield (
      StreamFactory.getDomainStreamSource(foldsDir + "/fold" + f + "_train.csv", domain, List.empty),
      StreamFactory.getDomainStreamSource(foldsDir + "/fold" + f + "_test.csv", domain, List.empty)
    )

    logger.info("Parsing pattern")
    val (formulas, exclusives, extras) = SREUtils.sre2formulas(patternFilePath, declarationsFilePath, withSelection = false)

    logger.info("Building SDFA")
    val t1 = System.nanoTime()
    val snfaStream = formulas.map(f => (SNFAUtils.buildSNFAForStream(f._1), f._2))
    val sdfa = snfaStream.map(s => (SFAUtils.determinizeI(s._1, exclusives, extras), s._2))
    sdfa.foreach(s => SDFAUtils.checkForDead(s._1))
    val sdfap = sdfa.map(s => (SDFAUtils.setPolicy(s._1, policy), s._2)).head
    val t2 = System.nanoTime()
    val detTime = t2 - t1
    logger.info("SDFA built")

    val stats: Stats = if (target.equalsIgnoreCase("ce")) {
      if (ForecastMethod.isClassification(spreadMethod)) new ClassificationStats(resultsDir, patternName)
      else new RegressionStats(resultsDir, patternName, folds.size)
    } else {
      new LogLossStats(resultsDir, patternName)
    }
    stats.writeHeader()

    logger.info("Running with SDFA")
    for (m <- sdfaOrders) {
      logger.info("Disambiguating")
      val t3 = System.nanoTime()
      val sdfaDis = SDFAUtils.disambiguateMutant(sdfap._1, m)
      val t4 = System.nanoTime()
      val disTime = t4 - t3
      logger.info("Disambiguation done")
      if (sdfaDis.size > maxSize) {
        logger.warn("Automaton size larger than allowed")
        val maxStatesReachedRow = stats.header.map(x => {
          if (x.equalsIgnoreCase("avgSize")) sdfaDis.size.toString
          else if (x.equalsIgnoreCase("extraTime")) (detTime + disTime).toString
          else "-1"
        })
        stats.writeRow(maxStatesReachedRow)
      } else {
        val sdfap = SDFAProvider(SDFASourceDirect(List(sdfaDis), List(formulas.head._3)))
        val sdfai = sdfap.provide()
        //val sdfaifn: String = resultsDir + "/sdfa_m" + m + ".fsm"
        //SerializationUtils.write2File[SDFAInterface](sdfai, sdfaifn)
        sdfai.foreach(f => f.estimateRemainingPercentage)
        val sdfaip = SDFAProvider(SDFASourceDirectI(sdfai))
        val fsmp = FSMProvider(sdfaip)
        // Training first
        val wtProviders = new Array[WtProvider](folds.size)
        val mcProviders = new Array[MarkovChainProvider](folds.size)
        val matrixTimes = new Array[Long](folds.size)
        val wtTimes = new Array[Long](folds.size)
        for (fold <- folds) {
          logger.info("Training for fold " + fold)
          val trainStreamSource = streamSources(fold - 1)._1
          val mct = MatrixMLETask(fsmp, trainStreamSource)
          val (mc, matrixTime) = mct.execute()
          matrixTimes(fold - 1) = matrixTime
          val wt1 = System.nanoTime()
          val mcp = MarkovChainProvider(MCSourceDirect(mc))
          mcProviders(fold - 1) = mcp
          val wtds = mc.head.computeWTDists(fsmp.provide().head, horizon, finalsEnabled)
          val wt2 = System.nanoTime()
          val wtTime = wt2 - wt1
          wtTimes(fold - 1) = wtTime
          val wtp = WtProvider(WtSourceDirect(List(wtds)))
          wtProviders(fold - 1) = wtp
        }
        val estimatedDistances = Utils.findDistances(fsmp)
        logger.info("Distances: " + estimatedDistances)
        if (target.equalsIgnoreCase("ce"))
          RunERF(m, folds, distances, maxSpreads, thresholds, stats, matrixTimes, wtTimes, wtProviders, fsmp, horizon,
            spreadMethod, streamSources, finalsEnabled, sdfaDis.size, detTime, disTime, maxSize, resultsDir)
        else
          RunLogLoss(m, folds, maxSize, stats, fsmp, mcProviders, streamSources, sdfaDis.size)
      }
    }
    stats.close()
  }

  private def RunLogLoss(
                          m: Int,
                          folds: List[Int],
                          maxSize: Int,
                          stats: Stats,
                          fsmp: FSMProvider,
                          mcProviders: Array[MarkovChainProvider],
                          streamSources: List[(StreamSource, StreamSource)],
                          sdfaDisSize: Int
                        ): Unit = {
    val rowPrefix = List(m.toString, maxSize.toString)
    logger.info("Running with m: " + m)
    for (fold <- folds) {
      logger.info("Fold " + fold)
      val mcp = mcProviders(fold - 1)
      val testStreamSource = streamSources(fold - 1)._2
      val testStream = testStreamSource.emitEventsAndClose(EmitMode.BUFFER)
      val llt = LogLossTask(fsmp, mcp, testStream)
      val prof = llt.execute()
      stats.update(prof, sdfaDisSize)
    }
    stats.writeResultsRow(rowPrefix)
    stats.reset()
    logger.info("Done with m: " + m)
  }

  private def RunERF(
                      m: Int,
                      folds: List[Int],
                      distances: List[(Double, Double)],
                      maxSpreads: List[Int],
                      thresholds: List[Double],
                      stats: Stats,
                      matrixTimes: Array[Long],
                      wtTimes: Array[Long],
                      wtProviders: Array[WtProvider],
                      fsmp: FSMProvider,
                      horizon: Int,
                      spreadMethod: ForecastMethod,
                      streamSources: List[(StreamSource, StreamSource)],
                      finalsEnabled: Boolean,
                      sdfaDisSize: Int,
                      detTime: Long,
                      disTime: Long,
                      maxSize: Int,
                      resultsDir: String
                    ): Unit = {
    for (
      distance <- distances;
      maxSpread <- maxSpreads;
      threshold <- thresholds
    ) {
      val rowPrefix = List(m.toString, maxSpread.toString, distance._1.toString, distance._2.toString, threshold.toString, maxSize.toString)
      stats.setModelTimes(matrixTimes.toList)
      stats.setWtTimes(wtTimes.toList)
      logger.info("Running with m/spread/dist/thres: " + m + "/" + maxSpread + "/" + distance + "/" + threshold)
      for (fold <- folds) {
        logger.info("Fold " + fold)
        val wtp = wtProviders(fold - 1)
        val pt = WtPredictorTask(fsmp, wtp, horizon, threshold, maxSpread, spreadMethod)
        val (pred, predTime) = pt.execute()
        //val wtifn: String = resultsDir + "/sdfa_WTI_m" + m + "_h" + horizon + "_t" + threshold + "_s" + maxSpread + "_m" + spreadMethod + ".wt"
        //SerializationUtils.write2File[WtInterface](pred, wtifn)
        val pp = ForecasterProvider(ForecasterSourceDirect(pred))
        val testStreamSource = streamSources(fold - 1)._2
        val erft = ERFTask(
          fsmp             = fsmp,
          pp               = pp,
          predictorEnabled = true,
          finalsEnabled    = finalsEnabled,
          expirationDeadline   = ConfigUtils.defaultExpiration,
          distance         = distance,
          streamSource     = testStreamSource,
          collectStats = true,
          show = false
        )
        val prof = erft.execute()
        stats.update(prof, sdfaDisSize, predTime, detTime + disTime)
      }
      stats.writeResultsRow(rowPrefix)
      stats.reset()
      logger.info("Done with m/spread/dist/thres/maxSize: " + m + "/" + maxSpread + "/" + distance + "/" + threshold + "/" + maxSize)
    }
  }

}
