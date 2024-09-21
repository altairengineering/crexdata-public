package ui.experiments.archived.caviar

import fsm.CountPolicy
import model.waitingTime.ForecastMethod

object ConfigurationsSPSAMeetingContext {
  val configSPSAFixed = CaviarConfigSPSA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/context.sre",
    patternName          = "meetingContextSPSAFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsContext.sre",
    orders               = List(1, 2, 3, 4),
    horizon              = 500,
    distances            = List((1, 1)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingContext/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    finalsEnabled        = true,
    policy               = CountPolicy.OVERLAP,
    maxNoStates          = List(1000)
  )

  val configSPSAMax = CaviarConfigSPSA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/context.sre",
    patternName          = "meetingContextSPSAMax",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsContext.sre",
    orders               = List(1, 2, 3, 4),
    horizon              = 500,
    distances            = List((-1, -1)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingContext/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    finalsEnabled        = true,
    policy               = CountPolicy.OVERLAP,
    maxNoStates          = List(1000)
  )

  val configs: List[CaviarConfigSPSA] = List(configSPSAMax, configSPSAFixed)

}
