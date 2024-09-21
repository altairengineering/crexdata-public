package ui.experiments.exposed2cli.cards.archived

import com.typesafe.scalalogging.LazyLogging
import ui.experiments.exposed2cli.cards.inc.ConfigsSPSAInc
import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.archived.ThroughputExperiments

object CardsThroughputExperiments extends LazyLogging {
  def main(args: Array[String]): Unit = {
    val sdfaConfigs: List[ConfigExp] = List.empty //ConfigsSDFAInc.configs
    val spsaConfigs: List[ConfigExp] = ConfigsSPSAInc.configs
    val spstConfigs: List[ConfigExp] = List.empty //ConfigsSPSTnc.configs
    val meanConfigs: List[ConfigExp] = List.empty //ConfigsMeanInc.configs
    val totalIterations = 100
    val warmupIterations = 10

    for (config <- sdfaConfigs) {
      logger.info("\n\n\tSDFA throughput experiments: \n" + config.toString)
      ThroughputExperiments.RunExperiment(
        fsmType          = "sdfa",
        patternName      = config.patternName,
        domain           = config.domain,
        resultsDir       = config.resultsDir,
        testFile         = config.foldsDir + "/fold1" + "_test.csv",
        sdfaOrders       = config.orders,
        horizon          = config.horizon,
        thresholds       = config.thresholds,
        maxSpreads       = config.maxSpreads,
        spreadMethod     = config.spreadMethod,
        distances        = config.distances,
        finalsEnabled    = config.finalsEnabled,
        totalIterations  = totalIterations,
        warmupIterations = warmupIterations
      )
    }

    for (config <- spsaConfigs) {
      logger.info("\n\n\tSPSA throughput experiments: \n" + config.toString)
      ThroughputExperiments.RunExperiment(
        fsmType          = "spsa",
        patternName      = config.patternName,
        domain           = config.domain,
        resultsDir       = config.resultsDir,
        testFile         = config.foldsDir + "/fold1" + "_test.csv",
        sdfaOrders       = config.orders,
        horizon          = config.horizon,
        thresholds       = config.thresholds,
        maxSpreads       = config.maxSpreads,
        spreadMethod     = config.spreadMethod,
        distances        = config.distances,
        finalsEnabled    = config.finalsEnabled,
        totalIterations  = totalIterations,
        warmupIterations = warmupIterations
      )
    }

    for (config <- spstConfigs) {
      logger.info("\n\n\tSPST throughput experiments: \n" + config.toString)
      ThroughputExperiments.RunExperiment(
        fsmType          = "spst",
        patternName      = config.patternName,
        domain           = config.domain,
        resultsDir       = config.resultsDir,
        testFile         = config.foldsDir + "/fold1" + "_test.csv",
        sdfaOrders       = config.orders,
        horizon          = config.horizon,
        thresholds       = config.thresholds,
        maxSpreads       = config.maxSpreads,
        spreadMethod     = config.spreadMethod,
        distances        = config.distances,
        finalsEnabled    = config.finalsEnabled,
        totalIterations  = totalIterations,
        warmupIterations = warmupIterations
      )
    }

    for (config <- meanConfigs) {
      logger.info("\n\n\tSPSA throughput experiments: \n" + config.toString)
      ThroughputExperiments.RunExperiment(
        fsmType          = "mean",
        patternName      = config.patternName,
        domain           = config.domain,
        resultsDir       = config.resultsDir,
        testFile         = config.foldsDir + "/fold1" + "_test.csv",
        sdfaOrders       = config.orders,
        horizon          = config.horizon,
        thresholds       = config.thresholds,
        maxSpreads       = config.maxSpreads,
        spreadMethod     = config.spreadMethod,
        distances        = config.distances,
        finalsEnabled    = config.finalsEnabled,
        totalIterations  = totalIterations,
        warmupIterations = warmupIterations
      )
    }

  }

}
