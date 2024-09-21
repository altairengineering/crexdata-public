package ui.experiments.archived.synthetic

import ui.WayebCLI

object RunTest {
  val home: String = System.getenv("WAYEB_HOME")
  val psaDir: String = System.getenv("PSADIR")
  val resultsDir: String = psaDir + "/results"
  val patternFile: String = home + "/patterns/psa/001.sre"
  val declarationsFile: String = home + "/patterns/psa/declarations.sre"
  val trainDatasetFilename: String = psaDir + "/generated/train.csv"
  val testDatasetFilename: String = psaDir + "/generated/test.csv"
  val threshold = 0.75
  val maxSpread = 5
  val horizon = 200

  def main(args: Array[String]): Unit = {
    runDis("test")
    runRec("test")
    runTrain("test")
    runTest("test")
  }

  private def runDis(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val argsTest: Array[String] = Array(
      "fsmDisambiguate",
      "--fsmType:symbolic",
      "--patterns:" + patternFile,
      "--declarations:" + declarationsFile,
      "--outputFsm:" + fsm
    )
    WayebCLI.main(argsTest)
  }

  private def runRec(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val stats = resultsDir + "/" + patternName + ".stats.rec"
    val argsTest: Array[String] = Array(
      "recognition",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + trainDatasetFilename,
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

  private def runTrain(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val mc = resultsDir + "/" + patternName + ".mc"
    val argsTrain: Array[String] = Array(
      "estimateMatrix",
      "--matrixEstimator:mle",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + trainDatasetFilename,
      "--outputMc:" + mc
    )
    WayebCLI.main(argsTrain)
  }

  private def runTest(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val stats = resultsDir + "/" + patternName + ".stats"
    val mc = resultsDir + "/" + patternName + ".mc"
    val argsTest: Array[String] = Array(
      "forecasting",
      "--threshold:" + threshold,
      "--maxSpread:" + maxSpread,
      "--horizon:" + horizon,
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--mc:" + mc,
      "--stream:" + testDatasetFilename,
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

}
