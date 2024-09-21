package ui.experiments.archived.caviar

import fsm.CountPolicy
import model.waitingTime.ForecastMethod

object ConfigurationsSDFAMeetingSituation {
  val configSDFAFixed = CaviarConfigSDFA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/situation.sre",
    patternName          = "meetingSituationSDFAFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsSituation.sre",
    orders               = List(1, 2),
    horizon              = 200,
    distances            = List((1, 1)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingSituation/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    finalsEnabled        = false,
    policy               = CountPolicy.NONOVERLAP
  )

  val configSDFAMax = CaviarConfigSDFA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/situation.sre",
    patternName          = "meetingSituationSDFAMax",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsSituation.sre",
    orders               = List(1), //orders = List(1, 2),
    horizon              = 200,
    distances            = List((-1, -1)),
    maxSpreads           = List(10), //maxSpreads = List(5, 10, 15),
    thresholds           = List(0.5), //thresholds = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingSituation/folds",
    folds                = List(1), //folds = List(1,2,3,4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    finalsEnabled        = false,
    policy               = CountPolicy.NONOVERLAP
  )

  val configs: List[CaviarConfigSDFA] = List(configSDFAMax) //,configSDFAFixed)
}
