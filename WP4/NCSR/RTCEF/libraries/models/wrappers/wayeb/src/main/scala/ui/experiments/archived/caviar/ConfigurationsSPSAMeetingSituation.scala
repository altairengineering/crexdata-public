package ui.experiments.archived.caviar

import fsm.CountPolicy
import model.waitingTime.ForecastMethod

object ConfigurationsSPSAMeetingSituation {
  val configSPSAFixed = CaviarConfigSPSA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/situation.sre",
    patternName          = "meetingSituationSPSAFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsSituation.sre",
    orders               = List(1, 2, 3, 4),
    horizon              = 200,
    distances            = List(1, 2, 3, 4, 5).map(x => Constants.sampling * x).map(y => (y.toDouble, y.toDouble)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingSituation/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    finalsEnabled        = false,
    policy               = CountPolicy.NONOVERLAP,
    maxNoStates          = List(1000)
  )

  val configSPSAMax = CaviarConfigSPSA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/situation.sre",
    patternName          = "meetingSituationSPSAMax",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsSituation.sre",
    orders               = List(2), //orders = List(1, 2, 3, 4),
    horizon              = 200,
    distances            = List((-1, -1)),
    maxSpreads           = List(10), //maxSpreads = List(5, 10, 15),
    thresholds           = List(0.1), //thresholds = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingSituation/folds",
    folds                = List(1), //folds = List(1,2,3,4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    finalsEnabled        = false,
    policy               = CountPolicy.NONOVERLAP,
    maxNoStates          = List(1000)
  )

  val configs: List[CaviarConfigSPSA] = List(configSPSAMax) //,configSPSAFixed)
}
