package ui.experiments.archived.text

import ui.experiments.exposed2cli.{ConfigExp, PatternExperimentsSDFA, PatternExperimentsSPSA}

object TextExperiments {
  def main(args: Array[String]): Unit = {
    val sdfaConfigs: List[ConfigExp] = ConfigsSDFA.configs
    val spsaConfigs: List[ConfigExp] = ConfigsSPSA.configs

    for (config <- sdfaConfigs) {
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
  }

}
