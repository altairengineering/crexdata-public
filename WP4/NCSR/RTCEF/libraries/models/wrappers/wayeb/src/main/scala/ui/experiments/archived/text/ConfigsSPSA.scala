package ui.experiments.archived.text

import ui.experiments.exposed2cli.ConfigExp

object ConfigsSPSA {
  val configSPSALogLoss = new ConfigExp(
    domain               = "text",
    patternFilePath      = Constants.wayebHome + "/patterns/text/pattern.sre",
    patternName          = "spaceSPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/text/declarations.sre",
    resultsDir           = Constants.dataDir + "/results",
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8),
    foldsDir             = Constants.dataDir + "/stream/folds",
    folds                = List(1), //,2,3,4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.001),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.05)
  )

  val configs: List[ConfigExp] = List(configSPSALogLoss)

}
