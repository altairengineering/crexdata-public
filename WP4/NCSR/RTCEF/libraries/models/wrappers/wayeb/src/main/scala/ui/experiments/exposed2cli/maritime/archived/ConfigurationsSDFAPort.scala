package ui.experiments.exposed2cli.maritime.archived

import com.github.tototoshi.csv.CSVReader
import model.waitingTime.ForecastMethod

object ConfigurationsSDFAPort {

  // Port pattern, all vessels
  val configSDFADistancesFixed = MaritimeConfigSDFA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSDFADistancesFixed",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    orders               = List(1, 2),
    horizon              = 500,
    distances            = List(1, 3, 5, 7, 9).map(x => ConfigMaritimeExperiments.intervalSec * x).map(y => (y.toDouble, y.toDouble)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "port" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD
  )

  val configSDFADistancesMax = MaritimeConfigSDFA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSDFADistancesMax",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    orders               = List(1, 2),
    horizon              = 500,
    distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "port" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN
  )

  val configSDFADistancesHeadingFixed = MaritimeConfigSDFA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSDFADistancesHeadingFixed",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    orders               = List(1, 2),
    horizon              = 500,
    distances            = List(1, 3, 5, 7, 9).map(x => ConfigMaritimeExperiments.intervalSec * x).map(y => (y.toDouble, y.toDouble)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "port" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD
  )

  val configSDFADistancesHeadingMax = MaritimeConfigSDFA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSDFADistancesHeadingMax",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
    orders               = List(1, 2),
    horizon              = 500,
    distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(5, 10, 15),
    thresholds           = List(0.25, 0.5, 0.75),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "port" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN
  )

  val configsAllVessels: List[MaritimeConfigSDFA] = List(
    configSDFADistancesMax,
    configSDFADistancesHeadingMax,
    configSDFADistancesFixed,
    configSDFADistancesHeadingFixed
  )

  // Port pattern, single vessels
  var configsPerVessel: List[MaritimeConfigSDFA] = List.empty
  val vesselsWithCEsPath: String = ConfigMaritimeExperiments.dataDir +
    "/enriched/" +
    ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
    "port/vesselsWithCEs.csv"
  val vesselsWithCEsReader: CSVReader = CSVReader.open(vesselsWithCEsPath)
  val vessels: List[(String, Int)] = vesselsWithCEsReader.all().map(x => (x.head, x(1).toInt)).sortBy((_._2)).reverse //.map(x => x.head)
  val vesselsMostCEs: List[String] = List(vessels.head._1)
  for (portVessel <- vesselsMostCEs) {
    val configSDFADistancesFixed = MaritimeConfigSDFA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSDFADistancesFixed" + portVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
      orders               = List(1, 2),
      horizon              = 500,
      distances            = List(1, 3, 5, 7, 9).map(x => ConfigMaritimeExperiments.intervalSec * x).map(y => (y.toDouble, y.toDouble)),
      maxSpreads           = List(5, 10, 15),
      thresholds           = List(0.25, 0.5, 0.75),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/" + portVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.FIXEDSPREAD
    )
    //configsPerVessel = configSDFADistancesFixed::configsPerVessel

    val configSDFADistancesMax = MaritimeConfigSDFA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSDFADistancesMax" + portVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
      orders               = List(1, 2),
      horizon              = 500,
      distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
      maxSpreads           = List(5, 10, 15),
      thresholds           = List(0.25, 0.5, 0.75),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/" + portVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.SMARTSCAN
    )
    //configsPerVessel = configSDFADistancesMax::configsPerVessel

    val configSDFADistancesHeadingFixed = MaritimeConfigSDFA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSDFADistancesHeadingFixed" + portVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
      orders               = List(1, 2),
      horizon              = 500,
      distances            = List(1, 3, 5, 7, 9).map(x => ConfigMaritimeExperiments.intervalSec * x).map(y => (y.toDouble, y.toDouble)),
      maxSpreads           = List(5, 10, 15),
      thresholds           = List(0.25, 0.5, 0.75),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/" + portVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.FIXEDSPREAD
    )
    //configsPerVessel = configSDFADistancesHeadingFixed::configsPerVessel

    val configSDFADistancesHeadingMax = MaritimeConfigSDFA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSDFADistancesHeadingMax" + portVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistanceHeading.sre",
      orders               = List(1, 2),
      horizon              = 500,
      distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
      maxSpreads           = List(5, 10, 15),
      thresholds           = List(0.25, 0.5, 0.75),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/" + portVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.SMARTSCAN
    )
    //configsPerVessel = configSDFADistancesHeadingMax::configsPerVessel

    configsPerVessel = List(configSDFADistancesMax, configSDFADistancesHeadingMax, configSDFADistancesFixed, configSDFADistancesHeadingFixed)
  }

  val configs: List[MaritimeConfigSDFA] =
    //configsPerVessel //:::
    configsAllVessels

}
