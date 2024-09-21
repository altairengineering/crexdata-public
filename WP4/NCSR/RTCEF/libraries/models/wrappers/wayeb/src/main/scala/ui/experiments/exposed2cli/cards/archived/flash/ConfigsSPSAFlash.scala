package ui.experiments.exposed2cli.cards.archived.flash

import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.cards.Constants
import model.waitingTime.ForecastMethod

object ConfigsSPSAFlash {
  val configSPSAFixed = new ConfigExp(
    domain               = "feedzai",
    patternFilePath      = Constants.wayebHome + "/patterns/feedzai/flash/pattern.sre",
    patternName          = "flashSPSAFixed",
    declarationsFilePath = "", //Constants.wayebHome + "/patterns/feedzai/flash/declarations.sre",
    resultsDir           = Constants.dataDir + "/results",
    orders               = List(1, 2, 3, 4, 5),
    horizon              = 200,
    distances            = List(1, 2, 3, 4, 5, 6, 7).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(0, 2, 4, 6),
    thresholds           = List(0.0, 0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/flash/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    maxNoStatesList      = List(1000)
  )

  val configSPSAMax = new ConfigExp(
    domain               = "feedzai",
    patternFilePath      = Constants.wayebHome + "/patterns/feedzai/flash/pattern.sre",
    patternName          = "flashSPSAMax",
    declarationsFilePath = "",
    resultsDir           = Constants.dataDir + "/results",
    orders               = List(1, 2, 3, 4, 5),
    horizon              = 200,
    maxSpreads           = List(0, 2, 4, 6),
    thresholds           = List(0.0, 0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/flash/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    maxNoStatesList      = List(1000)
  )

  val configs: List[ConfigExp] = List(configSPSAMax, configSPSAFixed)

}
