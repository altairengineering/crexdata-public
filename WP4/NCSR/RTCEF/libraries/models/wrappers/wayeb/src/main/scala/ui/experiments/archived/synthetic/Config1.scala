package ui.experiments.archived.synthetic

import ui.experiments.exposed2cli.ConfigExp

object Config1 {
  val configSDFAMax = ConfigSDFA(
    patternFilePath      = Constants.wayebHome + "/patterns/psa/001.sre",
    patternName          = "001SDFAMax",
    declarationsFilePath = Constants.wayebHome + "/patterns/psa/declarations.sre",
    orders               = List(1, 2),
    horizon              = 200,
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75)
  )

  val configs: List[ConfigSDFA] = List(configSDFAMax)
}
