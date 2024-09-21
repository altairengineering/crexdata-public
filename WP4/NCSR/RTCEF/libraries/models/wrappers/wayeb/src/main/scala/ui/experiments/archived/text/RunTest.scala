package ui.experiments.archived.text

import ui.WayebCLI

object RunTest {
  val home: String = System.getenv("WAYEB_HOME")
  val outDir: String = System.getenv("TEXTDIR")
  val domain: String = "text"
  val resultsDir: String = outDir + "/results"
  val trainDatasetFilename: String = outDir + "/stream/fold1_train.csv"
  val testDatasetFilename: String = outDir + "/stream/fold1_test.csv"
  val patternFile: String = home + "/patterns/text/pattern.sre"
  val declarationsFile: String = home + "/patterns/text/declarations.sre"
  //val declarationsFile: String = ""
  val policy: String = "nonoverlap"
  val finalsEnabled: String = "false"
  val threshold = 0.5
  val maxSpread = 5
  val horizon = 200
  val spreadMethod: String = "smart-scan"

  def main(args: Array[String]): Unit = {
    runDis("test")
    runRec("test")
    //runTrain("test")
    //runTest("test")
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
      "--domainSpecificStream:" + domain,
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
      "--domainSpecificStream:" + domain,
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
      "--domainSpecificStream:" + domain,
      "--streamArgs:",
      "--finalsEnabled:" + finalsEnabled,
      "--spreadMethod:" + spreadMethod,
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

}
