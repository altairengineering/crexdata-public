package ui.experiments.exposed2cli.cards.archived.dec

import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.cards.Constants
import model.waitingTime.ForecastMethod

object ConfigsSDFADec {
  val configSDFAFixed = new ConfigExp(
    domain               = "feedzai",
    patternFilePath      = Constants.wayebHome + "/patterns/feedzai/decreasing/pattern.sre",
    patternName          = "decreasingSDFAFixed",
    declarationsFilePath = "", //Constants.wayebHome + "/patterns/feedzai/decreasing/declarations.sre",
    resultsDir           = Constants.dataDir + "/results",
    orders               = List(1, 2, 3, 4, 5),
    horizon              = 200,
    distances            = List(1, 2, 3, 4, 5, 6, 7).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(0, 2, 4, 6),
    thresholds           = List(0.0, 0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/decreasing/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD
  )

  val configSDFAMax = new ConfigExp(
    domain               = "feedzai",
    patternFilePath      = Constants.wayebHome + "/patterns/feedzai/decreasing/pattern.sre",
    patternName          = "decreasingSDFAMax",
    declarationsFilePath = "",
    resultsDir           = Constants.dataDir + "/results",
    orders               = List(1, 2, 3, 4, 5),
    horizon              = 200,
    maxSpreads           = List(0, 2, 4, 6),
    thresholds           = List(0.0, 0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/decreasing/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN
  )

  val configs: List[ConfigExp] = List(configSDFAMax, configSDFAFixed)
}
