package ui.experiments.archived.synthetic

object ConfigSDFA {
  def apply(
      patternFilePath: String,
      patternName: String,
      declarationsFilePath: String,
      orders: List[Int],
      horizon: Int,
      maxSpreads: List[Int],
      thresholds: List[Double]
  ): ConfigSDFA =
    new ConfigSDFA(
      patternFilePath,
      patternName,
      declarationsFilePath,
      orders,
      horizon,
      maxSpreads,
      thresholds
    )
}

class ConfigSDFA(
    val patternFilePath: String,
    val patternName: String,
    val declarationsFilePath: String,
    val orders: List[Int],
    val horizon: Int,
    val maxSpreads: List[Int],
    val thresholds: List[Double]
) {

}
