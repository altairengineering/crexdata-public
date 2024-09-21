package ui.experiments.archived.caviar

import model.waitingTime.ForecastMethod.ForecastMethod
import fsm.CountPolicy.CountPolicy

object CaviarConfigSDFA {
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
      finalsEnabled: Boolean,
      policy: CountPolicy
  ): CaviarConfigSDFA =
    new CaviarConfigSDFA(
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
      finalsEnabled,
      policy
    )
}

class CaviarConfigSDFA(
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
    val finalsEnabled: Boolean,
    val policy: CountPolicy
) {

}
