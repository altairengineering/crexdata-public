package ui.experiments.exposed2cli

import com.typesafe.scalalogging.LazyLogging
import fsm.symbolic.sfa.SFAUtils
import fsm.symbolic.sfa.sdfa.{SDFA, SDFAUtils}
import fsm.symbolic.sfa.snfa.SNFAUtils
import fsm.symbolic.sre.SREUtils
import model.waitingTime.ForecastMethod.ForecastMethod
import stream.StreamFactory
import utils.SerializationUtils
import model.vmm.{Isomorphism, VMMUtils}
import model.vmm.VMMUtils.createIsomorphism
import model.vmm.pst.spsa.SymbolicPSA
import workflow.provider._
import fsm.CountPolicy.CountPolicy
import stream.source.{EmitMode, StreamSource}
import model.vmm.pst.{CSTLearner, PSTLearner}
import workflow.provider.source.matrix.MCSourceSPSA
import workflow.provider.source.forecaster.ForecasterSourceDirect
import workflow.provider.source.spsa.{SPSASourceDirect, SPSASourceDirectI}
import workflow.provider.source.wt.WtSourceDirect
import workflow.task.engineTask.ERFTask
import workflow.task.predictorTask.WtPredictorTask
import model.waitingTime.ForecastMethod
import breeze.stats.mean
import com.github.tototoshi.csv.CSVWriter
import fsm.SPSAInterface
import model.forecaster.WtInterface
import ui.ConfigUtils

object PatternExperimentsSPSA extends LazyLogging {

  def RunExperiments(
                      domain: String,
                      foldsDir: String,
                      folds: List[Int],
                      patternFilePath: String,
                      patternName: String,
                      declarationsFilePath: String,
                      resultsDir: String,
                      spsaOrders: List[Int],
                      horizon: Int,
                      finalsEnabled: Boolean,
                      distances: List[(Double, Double)],
                      maxSpreads: List[Int],
                      thresholds: List[Double],
                      spreadMethod: ForecastMethod,
                      policy: CountPolicy,
                      maxNoStatesList: List[Int],
                      pMins: List[Double],
                      alphas: List[Double],
                      gammas: List[Double],
                      rs: List[Double],
                      maxSize: Int,
                      target: String
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
    val iso = createIsomorphism(snfaStream.head._1, exclusives, extras)
    val t2 = System.nanoTime()
    val detTime = t2 - t1
    logger.info("SDFA built")
    val partitionAttribute = formulas.head._3

    if (target.equalsIgnoreCase("ce"))
      RunERF(
        folds              = folds,
        spreadMethod       = spreadMethod,
        resultsDir         = resultsDir,
        patternName        = patternName,
        maxNoStatesList    = maxNoStatesList,
        spsaOrders         = spsaOrders,
        pMins              = pMins,
        alphas             = alphas,
        gammas             = gammas,
        rs                 = rs,
        streamSources      = streamSources,
        sdfa               = sdfap._1,
        iso                = iso,
        partitionAttribute = partitionAttribute,
        horizon            = horizon,
        finalsEnabled      = finalsEnabled,
        distances          = distances,
        maxSpreads         = maxSpreads,
        thresholds         = thresholds,
        detTime            = detTime
      )
    else {
      val resultsFileName = resultsDir + "/" + patternName + ".csv"
      RunLogLoss(
        folds              = folds,
        iso                = iso,
        partitionAttribute = partitionAttribute,
        pMins              = pMins,
        alphas             = alphas,
        gammas             = gammas,
        rs                 = rs,
        maxOrders          = spsaOrders,
        streamSources      = streamSources,
        resultsFileName    = resultsFileName
      )
    }
  }

  private def RunERF(
                      folds: List[Int],
                      spreadMethod: ForecastMethod,
                      resultsDir: String,
                      patternName: String,
                      maxNoStatesList: List[Int],
                      spsaOrders: List[Int],
                      pMins: List[Double],
                      alphas: List[Double],
                      gammas: List[Double],
                      rs: List[Double],
                      streamSources: List[(StreamSource, StreamSource)],
                      sdfa: SDFA,
                      iso: Isomorphism,
                      partitionAttribute: String,
                      horizon: Int,
                      finalsEnabled: Boolean,
                      distances: List[(Double, Double)],
                      maxSpreads: List[Int],
                      thresholds: List[Double],
                      detTime: Long
                    ): Unit = {
    val stats: Stats =
      if (ForecastMethod.isClassification(spreadMethod)) new ClassificationStats(resultsDir, patternName)
      else new RegressionStats(resultsDir, patternName, folds.size)
    stats.writeHeader()
    logger.info("Running with SPSA")
    for (
      pMin <- pMins;
      alpha <- alphas;
      gamma <- gammas;
      r <- rs;
      m <- spsaOrders
    ) {
      val wtProviders = new Array[WtProvider](folds.size)
      val fsmProviders = new Array[FSMProvider](folds.size)
      val spsas = new Array[SymbolicPSA](folds.size)
      val vmmTimes = new Array[Long](folds.size)
      val wtTimes = new Array[Long](folds.size)
      for (fold <- folds) {
        logger.info("Training for fold " + fold)
        val trainStreamSource = streamSources(fold - 1)._1
        val vmmt1 = System.nanoTime()
        val (spsa, _) = VMMUtils.learnSPSAFromSingleSDFA(
          sdfa,
          m,
          trainStreamSource,
          iso,
          partitionAttribute,
          pMin, alpha, gamma, r
        )
        val vmmt2 = System.nanoTime()
        val vmmTime = vmmt2 - vmmt1
        val spsaProv = SPSAProvider(SPSASourceDirect(List(spsa), List(partitionAttribute)))
        val spsaI = spsaProv.provide()
        //val spsaifn: String = resultsDir + "/spsa_m" + m + ".fsm"
        //SerializationUtils.write2File[SPSAInterface](spsaI, spsaifn)
        spsaI.foreach(i => i.estimateRemainingPercentage)
        val spsaProvI = SPSAProvider(SPSASourceDirectI(spsaI))
        val fsmp = FSMProvider(spsaProvI)
        val mcp = MarkovChainProvider(MCSourceSPSA(fsmp))
        val mc = mcp.provide()
        val wt1 = System.nanoTime()
        val wtds = mc.head.computeWTDists(fsmp.provide().head, horizon, finalsEnabled)
        val wt2 = System.nanoTime()
        val wtTime = wt2 - wt1
        val wtp = WtProvider(WtSourceDirect(List(wtds)))
        wtProviders(fold - 1) = wtp
        fsmProviders(fold - 1) = fsmp
        spsas(fold - 1) = spsa
        vmmTimes(fold - 1) = vmmTime
        wtTimes(fold - 1) = wtTime
      }
      for (
        distance <- distances;
        maxSpread <- maxSpreads;
        threshold <- thresholds
      ) {
        val rowPrefix = List(m.toString, maxSpread.toString, distance._1.toString, distance._2.toString, threshold.toString, "-1")
        stats.setModelTimes(vmmTimes.toList)
        stats.setWtTimes(wtTimes.toList)
        logger.info("Running with m/spread/dist/thres: " + m + "/" + maxSpread + "/" + distance + "/" + threshold)
        for (fold <- folds) {
          logger.info("Fold " + fold)
          val wtp = wtProviders(fold - 1)
          val fsmp = fsmProviders(fold - 1)
          val testStreamSource = streamSources(fold - 1)._2
          val pt = WtPredictorTask(fsmp, wtp, horizon, threshold, maxSpread, spreadMethod)
          val (pred, predTime) = pt.execute()
          val wtifn: String = resultsDir + "/spsa_WTI_m" + m + "_h" + horizon + "_t" + threshold + "_s" + maxSpread + "_m" + spreadMethod + ".wt"
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
          stats.update(prof, spsas(fold - 1).getSize, predTime, detTime)
        }
        stats.writeResultsRow(rowPrefix)
        stats.reset()
        logger.info("Done with m/spread/dist/thres: " + m + "/" + maxSpread + "/" + distance + "/" + threshold)
      }
    }
    stats.close()
  }

  private def RunLogLoss(
                          folds: List[Int],
                          iso: Isomorphism,
                          partitionAttribute: String,
                          pMins: List[Double],
                          alphas: List[Double],
                          gammas: List[Double],
                          rs: List[Double],
                          maxOrders: List[Int],
                          streamSources: List[(StreamSource, StreamSource)],
                          resultsFileName: String
                        ): Unit = {
    val writer: CSVWriter = CSVWriter.open(resultsFileName, append = true)
    val header = List(
      "order",
      "pMin",
      "alpha",
      "gamma",
      "r",
      "avgLogLoss",
      "avgSize"
    )
    writer.writeRow(header)
    var bestLogLoss: Double = 10000.0
    var bestParValues: Map[String, String] = Map.empty
    for (
      pMin <- pMins;
      alpha <- alphas;
      gamma <- gammas;
      r <- rs;
      maxOrder <- maxOrders
    ) {
      val rowPrefix = List(maxOrder.toString, pMin.toString, alpha.toString, gamma.toString, r.toString)
      logger.info("Running with m/pMin/alpha/gamma/r: " + maxOrder + "/" + pMin + "/" + alpha + "/" + gamma + "/" + r)
      var lossSum: Double = 0.0
      var sizeSum: Int = 0
      for (fold <- folds) {
        logger.info("Fold " + fold)
        val trainStreamSource = streamSources(fold - 1)._1
        logger.info("Learning counter suffix tree")
        val cstLearner = new CSTLearner(maxOrder, iso, partitionAttribute)
        trainStreamSource.emitEventsToListener(cstLearner)
        val cst = cstLearner.getCST
        logger.info("Done with learning counter suffix tree")
        //logger.info(cst.toString)
        //logger.info("ISO: " + cst.getSymbols.map(s => (s, iso.getMinTermForSymbol(s))).toList)
        //cst.print(iso)
        logger.info("Learning prediction suffix tree")
        val symbols = iso.getSymbols
        val psTLearner = PSTLearner(symbols.toSet, maxOrder, pMin, alpha, gamma, r)
        val pst = psTLearner.learnVariant(cst, false)

        val testStreamSource = streamSources(fold - 1)._2
        val testStream = testStreamSource.emitEventsAndClose(EmitMode.BUFFER)
        val testStreams = testStream.partitionByAttribute(partitionAttribute)
        val testLists = testStreams.mapValues(s => s.getEventsAsList.reverse.map(x => iso.evaluate(x)))
        val losses = testLists.values.map(x => pst.avgLogLoss(x)).toList
        val avgLossForFold = mean(losses)
        lossSum += avgLossForFold
        sizeSum += pst.getSize.toInt
      }
      val avgLoss = lossSum / folds.size
      val avgSize = sizeSum / folds.size
      val row = rowPrefix ::: List(avgLoss.toString, avgSize.toString)
      if (avgLoss < bestLogLoss) {
        bestLogLoss = avgLoss
        bestParValues = header.zip(row).toMap
      }

      writer.writeRow(row)
      logger.info("avgLoss/avgSize: " + avgLoss + "/" + avgSize)
      logger.info("Done with m/pMin/alpha/gamma/r: " + maxOrder + "/" + pMin + "/" + alpha + "/" + gamma + "/" + r)
      logger.info("\n\n BEST THUS FAR: " + bestParValues + "\n\n")
    }
    logger.info("best: " + bestParValues)
    writer.close()
  }
}
