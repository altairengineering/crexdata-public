package ui.experiments.exposed2cli.maritime.port

import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.maritime.Constants
import model.waitingTime.ForecastMethod

object ConfigsHMMPort {
  val configClassificationSingleVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance1HMMClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    horizon              = 4,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(4),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configClassificationSingleVesselDistance3 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance3HMMClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance3.sre",
    resultsDir           = Constants.resultsDir,
    horizon              = 4,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(4),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configClassificationSingleVesselDistanceHeading = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistanceHeadingHMMClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    resultsDir           = Constants.resultsDir,
    horizon              = 4,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(4),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configClassificationSingleVesselNoFeatures = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselNoFeaturesHMMClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsNoFeatures.sre",
    resultsDir           = Constants.resultsDir,
    horizon              = 4,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(4),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configClassificationMultiVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portMultiVesselDistance1HMMClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    horizon              = 4,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(4),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches100/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK
  )

  val configRegressionSingleVesselNoFeatures = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselNoFeaturesHMMRegression",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsNoFeatures.sre",
    resultsDir           = Constants.resultsDir,
    horizon              = 5,
    distances            = List(1, 2, 3).map(x => (x.toDouble * 60, x.toDouble * 60)),
    maxSpreads           = List(0), //, 2, 4),
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
