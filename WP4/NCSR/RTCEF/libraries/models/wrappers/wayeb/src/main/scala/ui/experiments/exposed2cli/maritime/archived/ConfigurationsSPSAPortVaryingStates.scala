package ui.experiments.exposed2cli.maritime.archived

import com.github.tototoshi.csv.CSVReader
import model.waitingTime.ForecastMethod

object ConfigurationsSPSAPortVaryingStates {
  val configSPSADistancesMax = MaritimeConfigSPSA(
    patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
    patternName          = "portSPSADistancesMaxVaryingStates",
    declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
    orders               = List(1, 2, 3, 4),
    horizon              = 500,
    distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
    maxSpreads           = List(5),
    thresholds           = List(0.5),
    foldsDir             = ConfigMaritimeExperiments.dataDir +
      "/enriched/" +
      ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
      "port" + "/all/folds",
    folds                = List(1, 2, 3, 4),
    spreadMethod         = ForecastMethod.FIXEDSPREAD,
    maxNoStatesList      = List(50, 100, 500, 1000)
  )

  var configsPerVessel: List[MaritimeConfigSPSA] = List.empty
  val vesselsWithCEsPath: String = ConfigMaritimeExperiments.dataDir +
    "/enriched/" +
    ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
    "port/vesselsWithCEs.csv"
  val vesselsWithCEsReader: CSVReader = CSVReader.open(vesselsWithCEsPath)
  val vessels: List[(String, Int)] = vesselsWithCEsReader.all().map(x => (x.head, x(1).toInt)).sortBy((_._2)).reverse //.map(x => x.head)
  val vesselsMostCEs: List[String] = List(vessels.head._1)
  for (portVessel <- vesselsMostCEs) {
    val configSPSADistancesMax = MaritimeConfigSPSA(
      patternFilePath      = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/pattern.sre",
      patternName          = "portSPSADistancesMaxVaryingStates" + portVessel,
      declarationsFilePath = ConfigMaritimeExperiments.wayebHome + "/patterns/maritime/port/declarationsDistance1.sre",
      orders               = List(1, 2, 3, 4),
      horizon              = 500,
      distances            = List(-1).map(x => (x.toDouble, x.toDouble)),
      maxSpreads           = List(5),
      thresholds           = List(0.5),
      foldsDir             = ConfigMaritimeExperiments.dataDir +
        "/enriched/" +
        ConfigMaritimeExperiments.startTime + "-" + ConfigMaritimeExperiments.endTime + "_gap" + ConfigMaritimeExperiments.maxGap + "_interval" + ConfigMaritimeExperiments.intervalSec + "_speed" + ConfigMaritimeExperiments.speedThreshold + "/" +
        "port" + "/" + portVessel + "/folds",
      folds                = List(1, 2, 3, 4),
      spreadMethod         = ForecastMethod.SMARTSCAN,
      maxNoStatesList      = List(50, 100, 500, 1000)
    )
    configsPerVessel = configSPSADistancesMax :: configsPerVessel
  }

  val configs: List[MaritimeConfigSPSA] =
    configSPSADistancesMax ::
      configsPerVessel

}
