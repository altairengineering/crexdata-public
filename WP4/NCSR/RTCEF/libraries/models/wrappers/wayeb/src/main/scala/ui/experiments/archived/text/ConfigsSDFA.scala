package ui.experiments.archived.text

import ui.experiments.exposed2cli.ConfigExp

object ConfigsSDFA {
  val configSDFALogLoss = new ConfigExp(
    domain               = "text",
    patternFilePath      = Constants.wayebHome + "/patterns/text/pattern.sre",
    patternName          = "spaceSDFALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/text/declarations.sre",
    resultsDir           = Constants.dataDir + "/results",
    orders               = List(1, 2),
    horizon              = 200,
    foldsDir             = Constants.dataDir + "/stream/folds",
    folds                = List(1), //,2,3,4),
    target               = "sde",
    maxSize              = 10000
  )

  val configs: List[ConfigExp] = List(configSDFALogLoss) //configSDFAArgMax,configSDFAMax,configSDFAFixed,configSDFAClassify)

}
