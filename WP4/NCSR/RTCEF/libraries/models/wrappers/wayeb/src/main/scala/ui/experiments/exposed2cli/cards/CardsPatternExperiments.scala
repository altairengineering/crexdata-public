package ui.experiments.exposed2cli.cards

import java.nio.file.{Files, Paths}

import com.typesafe.scalalogging.LazyLogging
import ui.experiments.exposed2cli.cards.inc.{ConfigsHMMInc, ConfigsMeanInc, ConfigsSDFAInc, ConfigsSPSAInc, ConfigsSPSTnc}
import ui.experiments.exposed2cli.{ConfigExp, PatternExperimentsHMM, PatternExperimentsMean, PatternExperimentsSDFA, PatternExperimentsSPSA, PatternExperimentsSPST}

object CardsPatternExperiments extends LazyLogging {
  def main(args: Array[String]): Unit = {

    val resultsPath = Paths.get(Constants.wayebHome + "/results")
    val cardsResultsPath = Paths.get(Constants.resultsDir)
    if (!Files.exists(resultsPath) | !Files.isDirectory(resultsPath)) {
      Files.createDirectory(resultsPath)
      Files.createDirectory(cardsResultsPath)
    }
    else {
      if (!Files.exists(cardsResultsPath) | !Files.isDirectory(cardsResultsPath))
        Files.createDirectory(cardsResultsPath)
    }

    val sdfaConfigs: List[ConfigExp] = ConfigsSDFAInc.configs
    val spstConfigs: List[ConfigExp] = ConfigsSPSTnc.configs
    val meanConfigs: List[ConfigExp] = ConfigsMeanInc.configs
    val hmmConfigs: List[ConfigExp] = ConfigsHMMInc.configs
    val spsaConfigs: List[ConfigExp] = ConfigsSPSAInc.configs

    for (config <- meanConfigs) {
      logger.info("\n\n\tMEAN experiments: \n" + config.toString)
      PatternExperimentsMean.RunExperiments(
        domain               = config.domain,
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = config.resultsDir,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy
      )
    }

    for (config <- hmmConfigs) {
      logger.info("\n\n\tHMM experiments: \n" + config.toString)
      PatternExperimentsHMM.RunExperiments(
        domain               = config.domain,
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = config.resultsDir,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy
      )
    }

    for (config <- sdfaConfigs) {
      logger.info("\n\n\tSDFA experiments: \n" + config.toString)
      PatternExperimentsSDFA.RunExperiments(
        domain               = config.domain,
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = config.resultsDir,
        sdfaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxSize              = config.maxSize,
        target               = config.target
      )
    }

    for (config <- spsaConfigs) {
      logger.info("\n\n\tSPSA experiments: \n" + config.toString)
      PatternExperimentsSPSA.RunExperiments(
        domain               = config.domain,
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = config.resultsDir,
        spsaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxNoStatesList      = config.maxNoStatesList,
        maxSize              = config.maxSize,
        pMins                = config.pMins,
        alphas               = config.alphas,
        gammas               = config.gammas,
        rs                   = config.rs,
        target               = config.target
      )
    }

    for (config <- spstConfigs) {
      logger.info("\n\n\tSPST experiments: \n" + config.toString)
      PatternExperimentsSPST.RunExperiments(
        domain               = config.domain,
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = config.resultsDir,
        spstOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        pMins                = config.pMins,
        alphas               = config.alphas,
        gammas               = config.gammas,
        rs                   = config.rs,
        wt                   = config.wt,
        wtCutoffThreshold    = config.wtCutoffThreshold
      )
    }

  }
}
