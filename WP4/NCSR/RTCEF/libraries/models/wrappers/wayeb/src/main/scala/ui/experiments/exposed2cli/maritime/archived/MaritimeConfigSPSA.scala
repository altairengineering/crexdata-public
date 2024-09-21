package ui.experiments.exposed2cli.maritime.archived

import model.waitingTime.ForecastMethod.ForecastMethod

object MaritimeConfigSPSA {
  def apply(
      patternFilePath: String,
      patternName: String,
      declarationsFilePath: String,
      orders: List[Int],
      horizon: Int,
      distances: List[(Double, Double)],
      maxSpreads: List[Int],
      thresholds: List[Double],
      foldsDir: String,
      folds: List[Int],
      spreadMethod: ForecastMethod,
      maxNoStatesList: List[Int]
  ): MaritimeConfigSPSA =
    new MaritimeConfigSPSA(
      patternFilePath,
      patternName,
      declarationsFilePath,
      orders,
      horizon,
      distances,
      maxSpreads,
      thresholds,
      foldsDir,
      folds,
      spreadMethod,
      maxNoStatesList
    )
}

class MaritimeConfigSPSA(
    val patternFilePath: String,
    val patternName: String,
    val declarationsFilePath: String,
    val orders: List[Int],
    val horizon: Int,
    val distances: List[(Double, Double)],
    val maxSpreads: List[Int],
    val thresholds: List[Double],
    val foldsDir: String,
    val folds: List[Int],
    val spreadMethod: ForecastMethod,
    val maxNoStatesList: List[Int]
) {

}
