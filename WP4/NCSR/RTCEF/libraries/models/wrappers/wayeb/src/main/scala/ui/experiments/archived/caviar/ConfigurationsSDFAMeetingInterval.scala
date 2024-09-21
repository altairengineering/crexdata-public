package ui.experiments.archived.caviar

import fsm.CountPolicy
import ui.experiments.exposed2cli.ConfigExp
import model.waitingTime.ForecastMethod

object ConfigurationsSDFAMeetingInterval {
  /*val configSDFAFixed = CaviarConfigSDFA(
    patternFilePath = Constants.wayebHome + "/patterns/caviar/meeting/interval.sre",
    patternName = "meetingIntervalSDFAFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsInterval.sre",
    orders = List(1, 2),
    horizon = 500,
    distances = List(1),
    maxSpreads = List(5, 10, 15),
    thresholds = List(0.25, 0.5, 0.75),
    foldsDir = Constants.dataDir + "/enriched/meetingInterval/folds",
    folds = List(1,2,3,4),
    spreadMethod = ForecastMethod.FIXEDSPREAD,
    finalsEnabled = true,
    policy = CountPolicy.OVERLAP
  )*/

  val configSDFAFixed = new ConfigExp(
    domain               = "caviar",
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/interval.sre",
    patternName          = "meetingIntervalSDFAFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsInterval.sre",
    resultsDir           = Constants.dataDir + "/results",
    orders               = List(1, 2),
    horizon              = 500,
    distances            = List((1.0, 1.0)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingInterval/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    finalsEnabled        = true,
    policy               = CountPolicy.OVERLAP
  )

  /*val configSDFAMax = CaviarConfigSDFA(
    patternFilePath = Constants.wayebHome + "/patterns/caviar/meeting/interval.sre",
    patternName = "meetingIntervalSDFAMax",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsInterval.sre",
    orders = List(1, 2),
    horizon = 200,
    distances = List(-1),
    maxSpreads = List(5, 10, 15),
    thresholds = List(0.25, 0.5, 0.75),
    foldsDir = Constants.dataDir + "/enriched/meetingInterval/folds",
    folds = List(1,2,3,4),
    spreadMethod = ForecastMethod.SMARTSCAN,
    finalsEnabled = true,
    policy = CountPolicy.OVERLAP
  )*/

  val configSDFAMax = new ConfigExp(
    domain               = "caviar",
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/interval.sre",
    patternName          = "meetingIntervalSDFAMax",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsInterval.sre",
    resultsDir           = Constants.dataDir + "/results",
    orders               = List(1, 2),
    horizon              = 200,
    distances            = List((-1, -1)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingInterval/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    finalsEnabled        = true,
    policy               = CountPolicy.OVERLAP
  )

  val configs: List[ConfigExp] = List(configSDFAMax, configSDFAFixed)
}
