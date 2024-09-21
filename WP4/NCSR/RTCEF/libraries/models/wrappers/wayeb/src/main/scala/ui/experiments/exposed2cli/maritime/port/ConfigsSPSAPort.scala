package ui.experiments.exposed2cli.maritime.port

import ui.experiments.exposed2cli.ConfigExp
import ui.experiments.exposed2cli.maritime.Constants
import model.waitingTime.ForecastMethod

object ConfigsSPSAPort {

  val configLogLossSingleVesselDistanceHeading = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/patternLogLoss.sre",
    patternName          = "portSingleVesselDistanceHeadingSPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), //List(10)
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds                = List(1, 2, 3, 4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.001),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.03)
  )

  val configLogLossSingleVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/patternLogLoss.sre",
    patternName          = "portSingleVesselDistance1SPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), //List(10)
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds                = List(1, 2, 3, 4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.001),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.05)
  )

  val configLogLossSingleVesselDistance3 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/patternLogLoss.sre",
    patternName          = "portSingleVesselDistance3SPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance3.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), //List(10)
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds                = List(1, 2, 3, 4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.001),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.02)
  )

  val configLogLossSingleVesselNoFeatures = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/patternLogLoss.sre",
    patternName          = "portSingleVesselNoFeaturesSPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsNoFeatures.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), //List(10)
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds                = List(1, 2, 3, 4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.0001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.01),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.005)
  )

  val configLogLossMultiVesselDistanceHeading = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/patternLogLoss.sre",
    patternName          = "portMultiVesselDistanceHeadingSPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), //List(10)
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches100/port/all/folds",
    folds                = List(1, 2, 3, 4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.0001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.001),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.05)
  )

  val configLogLossMultiVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/patternLogLoss.sre",
    patternName          = "portMultiVesselDistance1SPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), //List(10)
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches100/port/all/folds",
    folds                = List(1, 2, 3, 4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.0001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.01),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.005)
  )

  val configLogLossMultiVesselDistance3 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/patternLogLoss.sre",
    patternName          = "portMultiVesselDistance3SPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance3.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), //List(10)
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches100/port/all/folds",
    folds                = List(1, 2, 3, 4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.0001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.01),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.005)
  )

  val configLogLossMultiVesselNoFeatures = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/patternLogLoss.sre",
    patternName          = "portMultiVesselNoFeaturesSPSALogLoss",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsNoFeatures.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), //List(10)
    foldsDir             = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches100/port/all/folds",
    folds                = List(1, 2, 3, 4),
    target               = "sde",
    pMins                = List(0.0001, 0.001, 0.01), //List(0.0001),
    alphas               = List(0.0), //List(0.0),
    gammas               = List(0.0001, 0.001, 0.01), //List(0.01),
    rs                   = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05) //List(1.005)
  )

  val configClassificationSingleVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance1SPSAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK,
    pMins        = List(0.001),
    alphas       = List(0.0),
    gammas       = List(0.001),
    rs           = List(1.05)
  )

  val configClassificationSingleVesselDistance3 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance3SPSAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance3.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK,
    pMins        = List(0.001),
    alphas       = List(0.0),
    gammas       = List(0.001),
    rs           = List(1.02)
  )

  val configClassificationSingleVesselDistanceHeading = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistanceHeadingSPSAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK,
    pMins        = List(0.001),
    alphas       = List(0.0),
    gammas       = List(0.001),
    rs           = List(1.03)
  )

  val configClassificationSingleVesselNoFeatures = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselNoFeaturesSPSAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsNoFeatures.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches360/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK,
    pMins        = List(0.0001),
    alphas       = List(0.0),
    gammas       = List(0.01),
    rs           = List(1.005)
  )

  val configClassificationMultiVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portMultiVesselDistance1SPSAClassification",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2, 3),
    horizon              = 10,
    //distances            = List((0.0, 0.35), (0.35, 0.7), (0.7, 1.0)),
    distances    = List((0.0, 0.5), (0.5, 1.0)),
    maxSpreads   = List(10),
    thresholds   = List(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    foldsDir     = Constants.dataDir + "/enriched/1443650401-1459461585_gap1800_interval60_speed1.0_matches100/port/all/folds",
    folds        = List(1, 2, 3, 4),
    spreadMethod = ForecastMethod.CLASSIFY_NEXTK,
    pMins        = List(0.0001),
    alphas       = List(0.0),
    gammas       = List(0.01),
    rs           = List(1.005)
  )

  val configRegressionSingleVesselDistance1 = new ConfigExp(
    domain               = "maritime",
    patternFilePath      = Constants.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSingleVesselDistance1SPSARegression",
    declarationsFilePath = Constants.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    resultsDir           = Constants.resultsDir,
    orders               = List(1, 2),
    horizon              = 200,
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
