package ui.experiments.exposed2cli

import com.typesafe.scalalogging.LazyLogging
import fsm.SPSTInterface
import fsm.symbolic.sfa.SFAUtils
import fsm.symbolic.sfa.sdfa.SDFAUtils
import fsm.symbolic.sfa.snfa.SNFAUtils
import fsm.symbolic.sre.SREUtils
import fsm.CountPolicy.CountPolicy
import stream.StreamFactory
import model.waitingTime.ForecastMethod
import model.waitingTime.ForecastMethod.ForecastMethod
import ui.ConfigUtils
import workflow.provider.source.forecaster.ForecasterSourceDirect
import workflow.provider.source.pst.PSTSourceLearner
import workflow.provider.source.sdfa.SDFASourceDirect
import workflow.provider.source.spst.SPSTSourceDirectI
import workflow.provider.source.wt.WtSourceDirect
import workflow.provider.{FSMProvider, ForecasterProvider, PSTProvider, SDFAProvider, SPSTProvider, WtProvider}
import workflow.task.engineTask.ERFTask
import workflow.task.predictorTask.WtPredictorTask

object PatternExperimentsSPST extends LazyLogging {

  def RunExperiments(
                      domain: String,
                      foldsDir: String,
                      folds: List[Int],
                      patternFilePath: String,
                      patternName: String,
                      policy: CountPolicy,
                      declarationsFilePath: String,
                      spreadMethod: ForecastMethod,
                      spstOrders: List[Int],
                      pMins: List[Double],
                      alphas: List[Double],
                      gammas: List[Double],
                      rs: List[Double],
                      horizon: Int,
                      distances: List[(Double, Double)],
                      finalsEnabled: Boolean,
                      maxSpreads: List[Int],
                      thresholds: List[Double],
                      resultsDir: String,
                      wt: String,
                      wtCutoffThreshold: Double
                    ): Unit = {

    logger.info("Creating stream sources for folds")

    val streamSources = for (f <- folds) yield (
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
    //val iso = createIsomorphism(snfaStream.head._1,exclusives,extras)
    val t2 = System.nanoTime()
    val detTime = t2 - t1
    logger.info("SDFA built")
    val partitionAttribute = formulas.head._3
    val sdfaProv = SDFAProvider(SDFASourceDirect(List(sdfap._1)))

    val stats: Stats =
      if (ForecastMethod.isClassification(spreadMethod)) new ClassificationStats(resultsDir, patternName)
      else new RegressionStats(resultsDir, patternName, folds.size)
    stats.writeHeader()
    logger.info("Running with PST")
    for (
      pMin <- pMins;
      alpha <- alphas;
      gamma <- gammas;
      r <- rs;
      m <- spstOrders
    ) {
      val wtProviders = new Array[WtProvider](folds.size)
      val fsmProviders = new Array[FSMProvider](folds.size)
      val spsts = new Array[SPSTInterface](folds.size)
      val pstTimes = new Array[Long](folds.size)
      val wtTimes = new Array[Long](folds.size)
      for (fold <- folds) {
        logger.info("Training for fold " + fold)
        val trainStreamSource = streamSources(fold - 1)._1
        val pstt1 = System.nanoTime()
        val pstProvider = PSTProvider(PSTSourceLearner(sdfaProv, trainStreamSource, m, pMin, alpha, gamma, r))
        val (pst, iso) = pstProvider.provide().head
        val pstt2 = System.nanoTime()
        val pstTime = pstt2 - pstt1
        val spsti = SPSTInterface(pst, sdfap._1, iso, m, 0, partitionAttribute)
        val spstProv = SPSTProvider(SPSTSourceDirectI(List(spsti)))
        //val spstifn: String = resultsDir + "/spst_m" + m + ".fsm"
        //SerializationUtils.write2File[SPSTInterface](List(spsti), spstifn)
        spsti.estimateRemainingPercentage
        val fsmp = FSMProvider(spstProv)
        logger.info("Computing waiting-time distributions")
        val wt1 = System.nanoTime()
        //val wtds = spsti.computeWtDistsExhaustive(horizon)
        //val wtds = spsti.computeWtDists(horizon)
        val wtds = if (wt.equalsIgnoreCase("pseudo")) spsti.computePseudoWtDists(horizon) else spsti.computeWtDistsOpt(horizon, wtCutoffThreshold)
        val wt2 = System.nanoTime()
        val wtTime = wt2 - wt1
        val wtp = WtProvider(WtSourceDirect(List(wtds)))
        wtProviders(fold - 1) = wtp
        fsmProviders(fold - 1) = fsmp
        spsts(fold - 1) = spsti
        pstTimes(fold - 1) = pstTime
        wtTimes(fold - 1) = wtTime
      }
      for (
        distance <- distances;
        maxSpread <- maxSpreads;
        threshold <- thresholds
      ) {
        val rowPrefix = List(m.toString, maxSpread.toString, distance._1.toString, distance._2.toString, threshold.toString, "-1")
        stats.setModelTimes(pstTimes.toList)
        stats.setWtTimes(wtTimes.toList)
        logger.info("Running with m/spread/dist/thres: " + m + "/" + maxSpread + "/" + distance + "/" + threshold)
        for (fold <- folds) {
          logger.info("Fold " + fold)
          val wtp = wtProviders(fold - 1)
          val fsmp = fsmProviders(fold - 1)
          val testStreamSource = streamSources(fold - 1)._2
          val pt = WtPredictorTask(fsmp, wtp, horizon, threshold, maxSpread, spreadMethod)
          val (pred, predTime) = pt.execute()
          //val wtifn: String = resultsDir + "/spst_WTI_m" + m + "_h" + horizon + "_t" + threshold + "_s" + maxSpread + "_m" + spreadMethod + ".wt"
          //SerializationUtils.write2File[WtInterface](pred, wtifn)
          val pp = ForecasterProvider(ForecasterSourceDirect(pred))
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
          stats.update(prof, spsts(fold - 1).getSize, predTime, detTime)
        }
        stats.writeResultsRow(rowPrefix)
        stats.reset()
        logger.info("Done with m/spread/dist/thres: " + m + "/" + maxSpread + "/" + distance + "/" + threshold)
      }
    }
    stats.close()

  }

}
