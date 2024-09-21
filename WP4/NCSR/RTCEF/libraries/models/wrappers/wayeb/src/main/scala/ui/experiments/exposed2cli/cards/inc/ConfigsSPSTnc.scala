package ui.experiments.exposed2cli.cards.inc

import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.cards.Constants
import model.waitingTime.ForecastMethod

object ConfigsSPSTnc {

  val configClassification = new ConfigExp(
    domain               = "cards",
    patternFilePath      = Constants.wayebHome + "/patterns/cards/increasing/pattern.sre",
    patternName          = "increasingSPSTClassifyNextK",
    declarationsFilePath = Constants.wayebHome + "/patterns/cards/increasing/declarations.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7),
    horizon              = 8,
    distances            = List((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)),
    maxSpreads           = List(8),
    thresholds           = List(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),//List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir             = Constants.dataDir + "/enriched/increasing/folds",
    folds                = List(1), //,2,3,4),
    spreadMethod         = ForecastMethod.CLASSIFY_NEXTK,
    pMins                = List(0.0001),
    alphas               = List(0.0),
    gammas               = List(0.001),
    rs                   = List(1.05),
    wtCutoffThreshold    = 0.001
  )

  val configFixed = new ConfigExp(
    domain               = "cards",
    patternFilePath      = Constants.wayebHome + "/patterns/cards/increasing/pattern.sre",
    patternName          = "increasingSPSTFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/cards/increasing/declarations.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5),
    horizon              = 8,
    distances            = List(1, 2, 3, 4, 5, 6).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(0), //, 2, 4, 6),
    thresholds           = List(0.0), //, 0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/increasing/folds",
    folds                = List(1), //2,3,4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    pMins                = List(0.0001),
    alphas               = List(0.0),
    gammas               = List(0.001),
    rs                   = List(1.05)
  )

  val configPseudoClassification = new ConfigExp(
    domain               = "cards",
    patternFilePath      = Constants.wayebHome + "/patterns/cards/increasing/pattern.sre",
    patternName          = "increasingG1H8SPSTPseudoClassifyNextK",
    declarationsFilePath = Constants.wayebHome + "/patterns/cards/increasing/declarations.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2),
    horizon              = 4,
    distances            = List((0.0, 0.2), (0.2, 0.4), (0.4, 0.6)), //(0.6,0.8),(0.8,1.0)),
    maxSpreads           = List(4),
    thresholds           = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0), //List(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),thresholds = List(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    foldsDir             = Constants.dataDir + "/enriched/increasingG1H8/folds",
    folds                = List(1), //,2,3,4),
    spreadMethod         = ForecastMethod.CLASSIFY_NEXTK,
    pMins                = List(0.0001),
    alphas               = List(0.0),
    gammas               = List(0.001),
    rs                   = List(1.05),
    wt                   = "pseudo"
  )

  val configs: List[ConfigExp] = List(configClassification)

}
