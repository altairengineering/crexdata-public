package ui.experiments.exposed2cli

import com.typesafe.scalalogging.LazyLogging
import fsm.symbolic.sfa.SFAUtils
import fsm.symbolic.sfa.sdfa.SDFAUtils
import fsm.symbolic.sfa.snfa.SNFAUtils
import fsm.symbolic.sre.SREUtils
import fsm.CountPolicy.CountPolicy
import stream.StreamFactory
import workflow.provider.source.forecaster.ForecasterSourceDirect
import workflow.provider.{FSMProvider, ForecasterProvider, SDFAProvider, WtProvider}
import workflow.provider.source.sdfa.SDFASourceDirect
import workflow.provider.source.wt.WtSourceDirect
import workflow.task.engineTask.ERFTask
import workflow.task.estimatorTask.MeanTask
import workflow.task.predictorTask.WtPredictorTask
import model.waitingTime.ForecastMethod
import model.waitingTime.ForecastMethod.ForecastMethod
import ui.ConfigUtils

object PatternExperimentsMean extends LazyLogging {

  def RunExperiments(
                      domain: String,
                      foldsDir: String,
                      folds: List[Int],
                      patternFilePath: String,
                      patternName: String,
                      declarationsFilePath: String,
                      resultsDir: String,
                      horizon: Int,
                      finalsEnabled: Boolean,
                      distances: List[(Double, Double)],
                      maxSpreads: List[Int],
                      thresholds: List[Double],
                      spreadMethod: ForecastMethod,
                      policy: CountPolicy
                    ): Unit = {

    logger.info("Creating stream sources for folds")

    val streamSources = for (f <- folds) yield (
      StreamFactory.getDomainStreamSource(foldsDir + "/fold" + f + "_train.csv", domain, List.empty),
      StreamFactory.getDomainStreamSource(foldsDir + "/fold" + f + "_test.csv", domain, List.empty)
    )

    logger.info("Parsing pattern")
    val (formulas, exclusives, extras) = SREUtils.sre2formulas(patternFilePath, declarationsFilePath, false)

    logger.info("Building SDFA")
    val t1 = System.nanoTime()
    val snfaStream = formulas.map(f => (SNFAUtils.buildSNFAForStream(f._1), f._2))
    val sdfa = snfaStream.map(s => (SFAUtils.determinizeI(s._1, exclusives, extras), s._2))
    sdfa.foreach(s => SDFAUtils.checkForDead(s._1))
    val sdfap = sdfa.map(s => (SDFAUtils.setPolicy(s._1, policy), s._2)).head
    val t2 = System.nanoTime()
    val detTime = t2 - t1
    logger.info("SDFA built")
    //logger.info(sdfap._1.toString)

    val stats: Stats =
      if (ForecastMethod.isClassification(spreadMethod)) new ClassificationStats(resultsDir, patternName)
      else new RegressionStats(resultsDir, patternName, folds.size)
    stats.writeHeader()

    logger.info("Running with SDFA")
    logger.info("Disambiguating")
    val t3 = System.nanoTime()
    val sdfaDis = SDFAUtils.disambiguateMutant(sdfap._1, 0)
    val t4 = System.nanoTime()
    val disTime = t4 - t3
    logger.info("Disambiguation done")
    logger.debug("SDFADIS: " + sdfaDis.toString)
    val sdfaprov = SDFAProvider(SDFASourceDirect(List(sdfaDis), List(formulas.head._3)))
    val fsmp = FSMProvider(sdfaprov)
    val sdfai = sdfaprov.provide()
    //val meanfn: String = resultsDir + "/mean_m0.fsm"
    //SerializationUtils.write2File[SDFAInterface](sdfai, meanfn)
    val wtProviders = new Array[WtProvider](folds.size)
    val meanTimes = new Array[Long](folds.size)
    val wtTimes = new Array[Long](folds.size)
    for (fold <- folds) {
      logger.info("Training for fold " + fold)
      val trainStreamSource = streamSources(fold - 1)._1
      val mt = MeanTask(fsmp, trainStreamSource)
      val (rt, meanTime) = mt.execute()
      meanTimes(fold - 1) = meanTime
      val wt1 = System.nanoTime()
      val wtds = rt.head.computeWTDists(fsmp.provide().head, horizon, finalsEnabled)
      val wt2 = System.nanoTime()
      val wtTime = wt2 - wt1
      wtTimes(fold - 1) = wtTime
      val wtp = WtProvider(WtSourceDirect(List(wtds)))
      wtProviders(fold - 1) = wtp
    }
    for (
      distance <- distances;
      maxSpread <- maxSpreads;
      threshold <- thresholds
    ) {
      val rowPrefix = List("-1", maxSpread.toString, distance._1.toString, distance._2.toString, threshold.toString, sdfaDis.size.toString)
      stats.setModelTimes(meanTimes.toList)
      stats.setWtTimes(wtTimes.toList)
      logger.info("Running with spread/dist/thres: " + maxSpread + "/" + distance + "/" + threshold)
      for (fold <- folds) {
        logger.info("Fold " + fold)
        val wtp = wtProviders(fold - 1)
        val pt = WtPredictorTask(fsmp, wtp, horizon, threshold, maxSpread, spreadMethod)
        val (pred, predTime) = pt.execute()
        val wtifn: String = resultsDir + "/mean_WTI_m" + 0 + "_h" + horizon + "_t" + threshold + "_s" + maxSpread + "_m" + spreadMethod + ".wt"
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
        stats.update(prof, sdfaDis.size, predTime, detTime + disTime)
      }
      stats.writeResultsRow(rowPrefix)
      stats.reset()
      logger.info("Done with spread/dist/thres: " + maxSpread + "/" + distance + "/" + threshold)
    }
    stats.close()
  }
}
