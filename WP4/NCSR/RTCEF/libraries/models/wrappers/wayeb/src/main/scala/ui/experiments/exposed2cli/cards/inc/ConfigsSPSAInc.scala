package ui.experiments.exposed2cli.cards.inc

import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.cards.Constants
import model.waitingTime.ForecastMethod

object ConfigsSPSAInc {
  
  val configLogLoss = new ConfigExp(
    domain               = "cards",
    patternFilePath      = Constants.wayebHome + "/patterns/cards/increasing/patternLogLoss.sre",
    patternName          = "increasingSPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/cards/increasing/declarationsLogLoss.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8),
    foldsDir             = Constants.dataDir + "/enriched/increasing/folds",
    folds                = List(1), //,2,3,4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.001),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.05)
  )

  val configClassification = new ConfigExp(
    domain               = "cards",
    patternFilePath      = Constants.wayebHome + "/patterns/cards/increasing/pattern.sre",
    patternName          = "increasingSPSAClassifyNextK",
    declarationsFilePath = Constants.wayebHome + "/patterns/cards/increasing/declarations.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4),
    horizon              = 8,
    distances            = List((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)), //List((0.14,0.15), (0.28,0.29), (0.42,0.43), (0.57,0.58), (0.71,0.72), (0.85,0.86)),
    maxSpreads           = List(8),
    thresholds           = List(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),//List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0), 
    foldsDir             = Constants.dataDir + "/enriched/increasing/folds",
    folds                = List(1), //,2,3,4),
    spreadMethod         = ForecastMethod.CLASSIFY_NEXTK,
    pMins                = List(0.0001),
    alphas               = List(0.0),
    gammas               = List(0.001),
    rs                   = List(1.05)
  )

  val configFixed = new ConfigExp(
    domain               = "cards",
    patternFilePath      = Constants.wayebHome + "/patterns/cards/increasing/pattern.sre",
    patternName          = "increasingSPSAFixed",
    declarationsFilePath = Constants.wayebHome + "/patterns/cards/increasing/declarations.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3),
    horizon              = 200,
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

  val configs: List[ConfigExp] = List(configClassification)

}
