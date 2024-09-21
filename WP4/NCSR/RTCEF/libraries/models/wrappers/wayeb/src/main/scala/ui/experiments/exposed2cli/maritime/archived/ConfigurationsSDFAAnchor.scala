package ui.experiments.exposed2cli.maritime.archived

import com.github.tototoshi.csv.CSVReader
import model.waitingTime.ForecastMethod

object ConfigurationsSDFAAnchor {
  // Anchor pattern, all vessels
  val configSDFASpeedsFixed = MaritimeConfigSDFA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/highSpeed/pattern.sre",
    patternName          = "speedSDFASpeedsFixed",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/highSpeed/declarationsSpeeds2.sre",
    orders               = List(1),
    horizon              = 200,
    distances            = List(1, 2, 3, 4, 5, 6).map(x => ConfigMaritimeExperiments.intervalSec * x).map(y => (y.toDouble, y.toDouble)),
    maxSpreads           = List(2, 4, 6, 8, 10, 20),
    thresholds           = List(0.1, 0.2, 0.3),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "speed" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD
  )

  val configSDFASpeedsMax = MaritimeConfigSDFA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/highSpeed/pattern.sre",
    patternName          = "speedSDFASpeedsMax",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/highSpeed/declarationsSpeeds2.sre",
    orders               = List(1),
    horizon              = 200,
    distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(2, 4, 6, 8, 10, 20),
    thresholds           = List(0.1, 0.2, 0.3),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "speed" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.SMARTSCAN
  )

  val configsAllVessels: List[MaritimeConfigSDFA] = List(
    //configSDFASpeedsFixed
    configSDFASpeedsMax
  )

  // Anchor pattern, single vessels
  var configsPerVessel: List[MaritimeConfigSDFA] = List.empty
  val vesselsWithCEsPath: String = ConfigMaritimeExperiments.dataDir +
    "/enriched/" +
    ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
    "speed/vesselsWithCEs.csv"
  val vesselsWithCEsReader: CSVReader = CSVReader.open(vesselsWithCEsPath)
  val vessels: List[(String, Int)] = vesselsWithCEsReader.all().map(x => (x.head, x(1).toInt)).sortBy((_._2)).reverse
  val vesselsMostCEs: List[String] = List(vessels.head._1)
  for (anchorVessel <- vesselsMostCEs) {
    /*val configSDFASpeedsFixed = MaritimeConfigSDFA(
      patternFilePath = ConfigExperiments.wayebHome + "/patterns/maritime/anchor/pattern.sre",
      patternName = "anchorSDFASpeedsFixed" + anchorVessel,
      declarationsFilePath = ConfigExperiments.wayebHome + "/patterns/maritime/anchor/declarationsSpeeds.sre",
      orders = List(1),
      horizon = 500,
      distances = List(60),
      maxSpreads = List(5),
      thresholds = List(0.5),
      foldsDir = ConfigExperiments.dataDir +
        "/enriched/" +
        ConfigExperiments.startTime + "-" + ConfigExperiments.endTime + "_gap" + ConfigExperiments.maxGap + "_interval" + ConfigExperiments.intervalSec + "_speed" + ConfigExperiments.speedThreshold + "/" +
        "anchor" + "/" + anchorVessel + "/folds",
      folds = List(1,2,3,4),
      spreadMethod = ForecastMethod.FIXEDSPREAD
    )
    configsPerVessel = configSDFASpeedsFixed :: configsPerVessel*/

    val configSDFASpeedsMax = MaritimeConfigSDFA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/anchor/pattern.sre",
      patternName          = "anchorSDFASpeedsMax" + anchorVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/anchor/declarationsSpeeds2.sre",
      orders               = List(1),
      horizon              = 500,
      distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
      maxSpreads           = List(10, 20, 30),
      thresholds           = List(0.25, 0.5, 0.75),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "anchor" + "/" + anchorVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.SMARTSCAN
    )
    configsPerVessel = configSDFASpeedsMax :: configsPerVessel
  }

  val configs: List[MaritimeConfigSDFA] =
    configsAllVessels //:::
  //configsPerVessel

}
