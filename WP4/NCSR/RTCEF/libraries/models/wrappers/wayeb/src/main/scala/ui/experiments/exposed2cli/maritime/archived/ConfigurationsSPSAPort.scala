package ui.experiments.exposed2cli.maritime.archived

import com.github.tototoshi.csv.CSVReader
import model.waitingTime.ForecastMethod

object ConfigurationsSPSAPort {
  // Port pattern, all vessels

  val configSPSADistancesFixed = MaritimeConfigSPSA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSPSADistancesFixed",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    orders               = List(1, 2, 3, 4),
    horizon              = 500,
    distances            = List(1, 3, 5, 7, 9).map(x => ConfigMaritimeExperiments.intervalSec * x).map(y => (y.toDouble, y.toDouble)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "port" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    maxNoStatesList      = List(1000)
  )

  val configSPSADistancesMax = MaritimeConfigSPSA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSPSADistancesMax",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    orders               = List(1, 2, 3, 4),
    horizon              = 500,
    distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "port" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    maxNoStatesList      = List(1000)
  )

  val configSPSADistancesHeadingFixed = MaritimeConfigSPSA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSPSADistancesHeadingFixed",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    orders               = List(1, 2, 3),
    horizon              = 500,
    distances            = List(1, 3, 5, 7, 9).map(x => ConfigMaritimeExperiments.intervalSec * x).map(y => (y.toDouble, y.toDouble)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "port" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    maxNoStatesList      = List(1000)
  )

  val configSPSADistancesHeadingMax = MaritimeConfigSPSA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSPSADistancesHeadingMax",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    orders               = List(1, 2, 3),
    horizon              = 500,
    distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "port" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN,
    maxNoStatesList      = List(1000)
  )

  val configsAllVessels: List[MaritimeConfigSPSA] = List(
    configSPSADistancesMax,
    configSPSADistancesHeadingMax,
    configSPSADistancesFixed,
    configSPSADistancesHeadingFixed
  )

  // Port pattern, single vessels
  var configsPerVessel: List[MaritimeConfigSPSA] = List.empty
  val vesselsWithCEsPath: String = ConfigMaritimeExperiments.dataDir +
    "/enriched/" +
    ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
    "port/vesselsWithCEs.csv"
  val vesselsWithCEsReader: CSVReader = CSVReader.open(vesselsWithCEsPath)
  val vessels: List[(String, Int)] = vesselsWithCEsReader.all().map(x => (x.head, x(1).toInt)).sortBy((_._2)).reverse //.map(x => x.head)
  val vesselsMostCEs: List[String] = List(vessels.head._1)
  for (portVessel <- vesselsMostCEs) {
    val configSPSADistancesFixed = MaritimeConfigSPSA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSPSADistancesFixed" + portVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
      orders               = List(1, 2, 3, 4),
      horizon              = 500,
      distances            = List(1, 3, 5, 7, 9).map(x => ConfigMaritimeExperiments.intervalSec * x).map(y => (y.toDouble, y.toDouble)),
      maxSpreads           = List(5, 10, 15),
      thresholds           = List(0.25, 0.5, 0.75),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/" + portVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.FIXEDSPREAD,
      maxNoStatesList      = List(1000)
    )
    //configsPerVessel = configSPSADistancesFixed::configsPerVessel

    val configSPSADistancesMax = MaritimeConfigSPSA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSPSADistancesMax" + portVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
      orders               = List(1, 2, 3, 4),
      horizon              = 500,
      distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
      maxSpreads           = List(5, 10, 15),
      thresholds           = List(0.25, 0.5, 0.75),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/" + portVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.SMARTSCAN,
      maxNoStatesList      = List(1000)
    )
    //configsPerVessel = configSPSADistancesMax::configsPerVessel

    val configSPSADistancesHeadingFixed = MaritimeConfigSPSA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSPSADistancesHeadingFixed" + portVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
      orders               = List(1, 2, 3),
      horizon              = 500,
      distances            = List(1, 3, 5, 7, 9).map(x => ConfigMaritimeExperiments.intervalSec * x).map(y => (y.toDouble, y.toDouble)),
      maxSpreads           = List(5, 10, 15),
      thresholds           = List(0.25, 0.5, 0.75),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/" + portVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.FIXEDSPREAD,
      maxNoStatesList      = List(1000)
    )
    //configsPerVessel = configSPSADistancesHeadingFixed::configsPerVessel

    val configSPSADistancesHeadingMax = MaritimeConfigSPSA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSPSADistancesHeadingMax" + portVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
      orders               = List(1, 2, 3),
      horizon              = 500,
      distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
      maxSpreads           = List(5, 10, 15),
      thresholds           = List(0.25, 0.5, 0.75),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/" + portVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.SMARTSCAN,
      maxNoStatesList      = List(1000)
    )
    //configsPerVessel = configSPSADistancesHeadingMax::configsPerVessel

    configsPerVessel = List(configSPSADistancesMax, configSPSADistancesHeadingMax, configSPSADistancesFixed, configSPSADistancesHeadingFixed)

  }

  val configs: List[MaritimeConfigSPSA] =
    ///configsPerVessel //:::
    configsAllVessels
}
