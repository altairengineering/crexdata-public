package ui.experiments.archived.caviar

import fsm.CountPolicy
import model.waitingTime.ForecastMethod

object ConfigurationsSDFAMeetingContext {
  val configSDFAFixed = CaviarConfigSDFA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/context.sre",
    patternName          = "meetingContextSDFAFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsContext.sre",
    orders               = List(1, 2),
    horizon              = 500,
    distances            = List((1.0, 1.0)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingContext/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    finalsEnabled        = true,
    policy               = CountPolicy.OVERLAP
  )

  val configSDFAMax = CaviarConfigSDFA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/context.sre",
    patternName          = "meetingContextSDFAMax",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsContext.sre",
    orders               = List(1, 2),
    horizon              = 200,
    distances            = List((-1.0, -1.0)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingContext/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    finalsEnabled        = true,
    policy               = CountPolicy.OVERLAP
  )

  val configs: List[CaviarConfigSDFA] = List(configSDFAMax, configSDFAFixed)

}
