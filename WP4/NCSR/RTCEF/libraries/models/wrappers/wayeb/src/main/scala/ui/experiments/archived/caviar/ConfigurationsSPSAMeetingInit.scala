package ui.experiments.archived.caviar

import fsm.CountPolicy
import model.waitingTime.ForecastMethod

object ConfigurationsSPSAMeetingInit {
  val configSPSAFixed = CaviarConfigSPSA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/initiation.sre",
    patternName          = "meetingInitSPSAFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsInitiation.sre",
    orders               = List(1, 2, 3, 4),
    horizon              = 500,
    distances            = List((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingInitiation/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    finalsEnabled        = false,
    policy               = CountPolicy.NONOVERLAP,
    maxNoStates          = List(1000)
  )

  val configSPSAMax = CaviarConfigSPSA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/initiation.sre",
    patternName          = "meetingInitSPSAMax",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsInitiation.sre",
    orders               = List(1, 2, 3, 4),
    horizon              = 500,
    distances            = List((-1, -1)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingInitiation/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    finalsEnabled        = false,
    policy               = CountPolicy.NONOVERLAP,
    maxNoStates          = List(1000)
  )

  val configs: List[CaviarConfigSPSA] = List(configSPSAMax, configSPSAFixed)

}
