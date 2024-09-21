package ui.experiments.exposed2cli.cards.archived

import ui.WayebCLI

object RunTest {
  val home: String = System.getenv("WAYEB_HOME")
  val dataDir: String = System.getenv("FEEDZAIDIR")
  val resultsDir: String = dataDir + "/results"
  val trainDatasetFilename: String = dataDir + "/enriched/increasing/folds/fold1_train.csv"
  val testDatasetFilename: String = dataDir + "/enriched/increasing/folds/fold1_test.csv"
  val patternFile: String = home + "/patterns/feedzai/increasing/pattern.sre"
  val declarationsFile: String = home + "/patterns/feedzai/increasing/declarations.sre"
  //val declarationsFile: String = ""
  val policy: String = "nonoverlap"
  val finalsEnabled: String = "false"
  val threshold = 0.0
  val maxSpread = 0
  val horizon = 200
  val spreadMethod: String = "argmax"

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
      "--countPolicy:" + policy,
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
      "--stream:" + testDatasetFilename,
      "--domainSpecificStream:feedzai",
      "--streamArgs:",
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
      "--domainSpecificStream:feedzai",
      "--streamArgs:",
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
      "--domainSpecificStream:feedzai",
      "--streamArgs:",
      "--finalsEnabled:" + finalsEnabled,
      "--spreadMethod:" + spreadMethod,
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

}
