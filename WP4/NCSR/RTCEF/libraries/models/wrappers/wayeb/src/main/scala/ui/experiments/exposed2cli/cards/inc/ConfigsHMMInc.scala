package ui.experiments.exposed2cli.cards.inc

import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.cards.Constants
import model.waitingTime.ForecastMethod

object ConfigsHMMInc {
  val configClassification = new ConfigExp(
    domain               = "cards",
    patternFilePath      = Constants.wayebHome + "/patterns/cards/increasing/pattern.sre",
    patternName          = "increasingHMMClassifyNextK",
    declarationsFilePath = "", //Constants.wayebHome + "/patterns/cards/increasing/declarations.sre",
    resultsDir           = Constants.resultsDir,
    horizon              = 4,
    distances            = List((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)), //List((0.14,0.15), (0.28,0.29), (0.42,0.43), (0.57,0.58), (0.71,0.72), (0.85,0.86)),
    maxSpreads           = List(4),
    thresholds           = List(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),//List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir             = Constants.dataDir + "/enriched/increasing/folds",
    folds                = List(1), //,2,3,4),
    spreadMethod         = ForecastMethod.CLASSIFY_NEXTK
  )

  val configFixed = new ConfigExp(
    domain               = "cards",
    patternFilePath      = Constants.wayebHome + "/patterns/cards/increasing/pattern.sre",
    patternName          = "increasingHMMFixed",
    declarationsFilePath = "", //Constants.wayebHome + "/patterns/cards/increasing/declarations.sre",
    resultsDir           = Constants.resultsDir,
    horizon              = 4,
    distances            = List(1, 2, 3, 4, 5, 6).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(0), //, 2, 4),
    thresholds           = List(0.0), //, 0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/increasing/folds",
    folds                = List(1), //2,3,4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD
  )

  val configs: List[ConfigExp] = List(configClassification)

}
