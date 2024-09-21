package ui.experiments.exposed2cli.maritime.port

import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.maritime.Constants
import model.waitingTime.ForecastMethod

object ConfigsSPSTPort {
  val configClassificationSingleVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance1SPSTClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances         = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads        = List(10),
    thresholds        = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir          = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds             = List(1, 2, 3, 4),
    spreadMethod      = ForecastMethod.CLASSIFY_NEXTK,
    pMins             = List(0.001),
    alphas            = List(0.0),
    gammas            = List(0.001),
    rs                = List(1.05),
    wtCutoffThreshold = 0.001
  )

  val configClassificationSingleVesselDistance3 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance3SPSTClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance3.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances         = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads        = List(10),
    thresholds        = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir          = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds             = List(1, 2, 3, 4),
    spreadMethod      = ForecastMethod.CLASSIFY_NEXTK,
    pMins             = List(0.001),
    alphas            = List(0.0),
    gammas            = List(0.001),
    rs                = List(1.02),
    wtCutoffThreshold = 0.001
  )

  val configClassificationSingleVesselDistanceHeading = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistanceHeadingSPSTClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances         = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads        = List(10),
    thresholds        = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir          = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds             = List(1, 2, 3, 4),
    spreadMethod      = ForecastMethod.CLASSIFY_NEXTK,
    pMins             = List(0.001),
    alphas            = List(0.0),
    gammas            = List(0.001),
    rs                = List(1.03),
    wtCutoffThreshold = 0.001
  )

  val configClassificationSingleVesselNoFeatures = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselNoFeaturesSPSTClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsNoFeatures.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances         = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads        = List(10),
    thresholds        = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir          = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds             = List(1, 2, 3, 4),
    spreadMethod      = ForecastMethod.CLASSIFY_NEXTK,
    pMins             = List(0.0001),
    alphas            = List(0.0),
    gammas            = List(0.01),
    rs                = List(1.005),
    wtCutoffThreshold = 0.001
  )

  val configClassificationMultiVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portMultiVesselDistance1SPSTClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances         = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads        = List(10),
    thresholds        = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir          = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches100/port/all/folds",
    folds             = List(1, 2, 3, 4),
    spreadMethod      = ForecastMethod.CLASSIFY_NEXTK,
    pMins             = List(0.0001),
    alphas            = List(0.0),
    gammas            = List(0.01),
    rs                = List(1.005),
    wtCutoffThreshold = 0.001
  )

  val configRegressionSingleVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance1SPSTRegression",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6),
    horizon              = 10,
    distances            = List(1, 2, 3).map(x => (x.toDouble * 60, x.toDouble * 60)),
    maxSpreads           = List(0), //, 2, 4, 6),
    thresholds           = List(0.0), //, 0.25, 0.5, 0.75),
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    pMins                = List(0.001),
    alphas               = List(0.0),
    gammas               = List(0.001),
    rs                   = List(1.05)
  )

  val configs: List[ConfigExp] = List(
    configClassificationSingleVesselDistance1,
    configClassificationSingleVesselDistance3,
    configClassificationSingleVesselDistanceHeading,
    configClassificationSingleVesselNoFeatures,
    configClassificationMultiVesselDistance1
  )
}
