package ui.experiments.exposed2cli.maritime.archived

import model.waitingTime.ForecastMethod
import ui.ConfigUtils
import ui.experiments.exposed2cli.PatternExperimentsSPSA

object Testing {
  def main(args: Array[String]): Unit = {
    val configSPSADistancesMax = MaritimeConfigSPSA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSPSADistancesMax",
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
      orders               = List(1, 2, 3),
      horizon              = 500,
      distances            = List((-1.0, -1.0)),
      maxSpreads           = List(5),
      thresholds           = List(0.5),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/227592820/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.SMARTSCAN,
      maxNoStatesList      = List(1000)
    )

    PatternExperimentsSPSA.RunExperiments(
      domain               = "maritime",
      foldsDir             = configSPSADistancesMax.foldsDir,
      folds                = configSPSADistancesMax.folds,
      patternFilePath      = configSPSADistancesMax.patternFilePath,
      patternName          = configSPSADistancesMax.patternName,
      declarationsFilePath = configSPSADistancesMax.declarationsFilePath,
      resultsDir           = ConfigMaritimeExperiments.resultsDir,
      spsaOrders           = configSPSADistancesMax.orders,
      horizon              = configSPSADistancesMax.horizon,
      finalsEnabled        = ConfigMaritimeExperiments.finalsEnabled,
      distances            = configSPSADistancesMax.distances,
      maxSpreads           = configSPSADistancesMax.maxSpreads,
      thresholds           = configSPSADistancesMax.thresholds,
      spreadMethod         = configSPSADistancesMax.spreadMethod,
      policy               = ConfigUtils.defaultPolicy,
      maxNoStatesList      = configSPSADistancesMax.maxNoStatesList,
      pMins                = List(0.0),
      alphas               = List(0.0),
      gammas               = List(0.0),
      rs                   = List(0.0),
      target               = "ce",
      maxSize              = 2000
    )

    /*val configSDFADistancesHeadingMax = MaritimeConfigSDFA(
      patternFilePath = ConfigExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName = "portSDFADistancesHeadingMax",
      declarationsFilePath = ConfigExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
      orders = List(2),
      horizon = 500,
      distances = List(-1),
      maxSpreads = List(5),
      thresholds = List(0.5),
      foldsDir = ConfigExperiments.dataDir +
        "/enriched/" +
        ConfigExperiments.startTime + "-" + ConfigExperiments.endTime + "_gap" + ConfigExperiments.maxGap + "_interval" + ConfigExperiments.intervalSec + "_speed" + ConfigExperiments.speedThreshold + "/" +
        "port" + "/227592820/folds",
      folds = List(1,2,3,4),
      spreadMethod = ForecastMethod.SMARTSCAN
    )

    PatternExperimentsSDFA.RunExperiments(
      foldsDir = configSDFADistancesHeadingMax.foldsDir,
      folds = configSDFADistancesHeadingMax.folds,
      patternFilePath = configSDFADistancesHeadingMax.patternFilePath,
      patternName = configSDFADistancesHeadingMax.patternName,
      declarationsFilePath = configSDFADistancesHeadingMax.declarationsFilePath,
      sdfaOrders = configSDFADistancesHeadingMax.orders,
      horizon = configSDFADistancesHeadingMax.horizon,
      finalsEnabled = ConfigExperiments.finalsEnabled,
      distances = configSDFADistancesHeadingMax.distances,
      maxSpreads = configSDFADistancesHeadingMax.maxSpreads,
      thresholds = configSDFADistancesHeadingMax.thresholds,
      spreadMethod = configSDFADistancesHeadingMax.spreadMethod
    )*/

  }

}
