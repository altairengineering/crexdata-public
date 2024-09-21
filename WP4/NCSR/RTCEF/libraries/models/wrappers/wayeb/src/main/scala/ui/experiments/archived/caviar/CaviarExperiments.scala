package ui.experiments.archived.caviar

import ui.experiments.exposed2cli.{PatternExperimentsSDFA, PatternExperimentsSPSA}

object CaviarExperiments {
  def main(args: Array[String]): Unit = {
    val sdfaConfigsMeetingInterval: List[CaviarConfigSDFA] = List.empty //ConfigurationsSDFAMeetingInterval.configs
    val spsaConfigsMeetingInterval: List[CaviarConfigSPSA] = List.empty //ConfigurationsSPSAMeetingInterval.configs
    val sdfaConfigsMeetingInit: List[CaviarConfigSDFA] = List.empty //ConfigurationsSDFAMeetingInit.configs
    val spsaConfigsMeetingInit: List[CaviarConfigSPSA] = List.empty //ConfigurationsSPSAMeetingInit.configs
    val sdfaConfigsMeetingContext: List[CaviarConfigSDFA] = List.empty //ConfigurationsSDFAMeetingContext.configs
    val spsaConfigsMeetingContext: List[CaviarConfigSPSA] = List.empty //ConfigurationsSPSAMeetingContext.configs
    val sdfaConfigsMeetingSituation: List[CaviarConfigSDFA] = List.empty //ConfigurationsSDFAMeetingSituation.configs
    val spsaConfigsMeetingSituation: List[CaviarConfigSPSA] = ConfigurationsSPSAMeetingSituation.configs

    for (config <- sdfaConfigsMeetingInterval) {
      PatternExperimentsSDFA.RunExperiments(
        domain               = "caviar",
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = Constants.resultsDir,
        sdfaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxSize              = 2000,
        target               = "ce"
      )
    }

    for (config <- spsaConfigsMeetingInterval) {
      PatternExperimentsSPSA.RunExperiments(
        domain               = "caviar",
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = Constants.resultsDir,
        spsaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxNoStatesList      = config.maxNoStates,
        pMins                = List(0.0),
        alphas               = List(0.0),
        gammas               = List(0.0),
        rs                   = List(0.0),
        target               = "ce",
        maxSize              = 2000
      )
    }

    for (config <- sdfaConfigsMeetingInit) {
      PatternExperimentsSDFA.RunExperiments(
        domain               = "caviar",
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = Constants.resultsDir,
        sdfaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxSize              = 2000,
        target               = "ce"
      )
    }

    for (config <- spsaConfigsMeetingInit) {
      PatternExperimentsSPSA.RunExperiments(
        domain               = "caviar",
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = Constants.resultsDir,
        spsaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxNoStatesList      = config.maxNoStates,
        pMins                = List(0.0),
        alphas               = List(0.0),
        gammas               = List(0.0),
        rs                   = List(0.0),
        target               = "ce",
        maxSize              = 2000
      )
    }

    for (config <- sdfaConfigsMeetingContext) {
      PatternExperimentsSDFA.RunExperiments(
        domain               = "caviar",
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = Constants.resultsDir,
        sdfaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxSize              = 2000,
        target               = "ce"
      )
    }

    for (config <- spsaConfigsMeetingContext) {
      PatternExperimentsSPSA.RunExperiments(
        domain               = "caviar",
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = Constants.resultsDir,
        spsaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxNoStatesList      = config.maxNoStates,
        pMins                = List(0.0),
        alphas               = List(0.0),
        gammas               = List(0.0),
        rs                   = List(0.0),
        target               = "ce",
        maxSize              = 2000
      )
    }

    for (config <- sdfaConfigsMeetingSituation) {
      PatternExperimentsSDFA.RunExperiments(
        domain               = "caviar",
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = Constants.resultsDir,
        sdfaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxSize              = 2000,
        target               = "ce"
      )
    }

    for (config <- spsaConfigsMeetingSituation) {
      PatternExperimentsSPSA.RunExperiments(
        domain               = "caviar",
        foldsDir             = config.foldsDir,
        folds                = config.folds,
        patternFilePath      = config.patternFilePath,
        patternName          = config.patternName,
        declarationsFilePath = config.declarationsFilePath,
        resultsDir           = Constants.resultsDir,
        spsaOrders           = config.orders,
        horizon              = config.horizon,
        finalsEnabled        = config.finalsEnabled,
        distances            = config.distances,
        maxSpreads           = config.maxSpreads,
        thresholds           = config.thresholds,
        spreadMethod         = config.spreadMethod,
        policy               = config.policy,
        maxNoStatesList      = config.maxNoStates,
        pMins                = List(0.0),
        alphas               = List(0.0),
        gammas               = List(0.0),
        rs                   = List(0.0),
        target               = "ce",
        maxSize              = 2000
      )
    }

  }
}
