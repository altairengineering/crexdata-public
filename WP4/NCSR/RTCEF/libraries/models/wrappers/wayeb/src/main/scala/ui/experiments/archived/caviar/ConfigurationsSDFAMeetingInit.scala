package ui.experiments.archived.caviar

import fsm.CountPolicy
import model.waitingTime.ForecastMethod

object ConfigurationsSDFAMeetingInit {
  val configSDFAFixed = CaviarConfigSDFA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/initiation.sre",
    patternName          = "meetingInitSDFAFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsInitiation.sre",
    orders               = List(1, 2),
    horizon              = 500,
    distances            = List((1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0), (6.0, 6.0), (7.0, 7.0), (8.0, 8.0), (9.0, 9.0), (10.0, 10.0)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingInitiation/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    finalsEnabled        = false,
    policy               = CountPolicy.NONOVERLAP
  )

  val configSDFAMax = CaviarConfigSDFA(
    patternFilePath      = Constants.wayebHome + "/patterns/caviar/meeting/initiation.sre",
    patternName          = "meetingInitSDFAMax",
    declarationsFilePath = Constants.wayebHome + "/patterns/caviar/meeting/declarationsInitiation.sre",
    orders               = List(1, 2),
    horizon              = 200,
    distances            = List((-1.0, -1.0)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/meetingInitiation/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    finalsEnabled        = false,
    policy               = CountPolicy.NONOVERLAP
  )

  val configs: List[CaviarConfigSDFA] = List(configSDFAMax, configSDFAFixed)

}
