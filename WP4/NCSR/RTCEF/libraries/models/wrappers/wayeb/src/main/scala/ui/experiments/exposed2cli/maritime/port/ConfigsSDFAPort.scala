package ui.experiments.exposed2cli.maritime.port

import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.maritime.Constants
import model.waitingTime.ForecastMethod

object ConfigsSDFAPort {

  val configLogLossSingleVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/patternLogLoss.sre",
    patternName          = "portSingleVesselDistance1SDFALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3),
    horizon              = 200,
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds                = List(1, 2, 3, 4),
    target               = "sde",
    maxSize              = 10000
  )

  val configClassificationSingleVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance1SDFAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(0, 1, 2),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)), //List((0.33,0.34), (0.66,0.67), (1.0,1.0))
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configClassificationSingleVesselDistance3 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance3SDFAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance3.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(0, 1, 2),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configClassificationSingleVesselDistanceHeading = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistanceHeadingSDFAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(0, 1),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configClassificationSingleVesselNoFeatures = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselNoFeaturesSDFAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsNoFeatures.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(0, 1, 2),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configClassificationMultiVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portMultiVesselDistance1SDFAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(0, 1, 2),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches100/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configRegressionSingleVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance1SDFARegression",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(0, 1, 2),
    horizon              = 200,
    distances            = List(1, 2, 3).map(x => (x.toDouble * 60, x.toDouble * 60)),
    maxSpreads           = List(0), //, 2, 4, 6),
    thresholds           = List(0.0), //, 0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD
  )

  val configs: List[ConfigExp] = List(
    configClassificationSingleVesselDistance1,
    configClassificationSingleVesselDistance3,
    configClassificationSingleVesselDistanceHeading,
    configClassificationSingleVesselNoFeatures,
    configClassificationMultiVesselDistance1
  )
}
