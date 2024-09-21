package scripts.data.json

import ui.WayebCLI

object JsonTest {

  val home: String = System.getenv("WAYEB_HOME")
  val resultsDir: String = home + "/results/vodafone"
  val dataDir: String = home + "/scripts/data/tracknknow"

  def main(args: Array[String]): Unit = {
    runDis("test")
    runRec("test")
    //runTrain("test")
    //runTest("test")
  }

  private def runDis(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val patternFile = home + "/patterns/vodafone/testPattern.sre"
    val declarationsFile = home + "/patterns/vodafone/testDeclarations.sre"
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
    val testSet = dataDir + "/jsonSample.json"
    val stats = resultsDir + "/" + patternName + ".stats.rec"
    val argsTest: Array[String] = Array(
      "recognition",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + testSet,
      "--domainSpecificStream:json",
      "--streamArgs:",
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

  private def runTrain(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val trainSet = dataDir + "/jsonSample.json"
    val mc = resultsDir + "/" + patternName + ".mc"
    val argsTrain: Array[String] = Array(
      "estimateMatrix",
      "--matrixEstimator:mle",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + trainSet,
      "--domainSpecificStream:json",
      "--streamArgs:",
      "--outputMc:" + mc
    )
    WayebCLI.main(argsTrain)
  }

  private def runTest(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val testSet = dataDir + "/jsonSample.json"
    val stats = resultsDir + "/" + patternName + ".stats"
    val mc = resultsDir + "/" + patternName + ".mc"
    val threshold = 0.5
    val maxSpread = 50
    val horizon = 200
    val argsTest: Array[String] = Array(
      "forecasting",
      "--threshold:" + threshold,
      "--maxSpread:" + maxSpread,
      "--horizon:" + horizon,
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--mc:" + mc,
      "--stream:" + testSet,
      "--domainSpecificStream:json",
      "--streamArgs:",
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

}
