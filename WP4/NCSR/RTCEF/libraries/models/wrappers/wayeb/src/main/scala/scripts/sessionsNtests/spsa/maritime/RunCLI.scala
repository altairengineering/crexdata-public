package scripts.sessionsNtests.spsa.maritime

import ui.WayebCLI

object RunCLI {
  val home: String = System.getenv("WAYEB_HOME")

  def main(args: Array[String]): Unit = {
    runDis("test")
    runRec("test")
    //runTrainFMM("test")
    //runTestFMM("test")
  }

  private def runDis(patternName: String): Unit = {
    val fsm = ConfigCLI.resultsDir + "/" + patternName + ".fsm"
    val patternFile = ConfigCLI.patternFile
    val declarationsFile = ConfigCLI.declarationsFile
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
    val fsm = ConfigCLI.resultsDir + "/" + patternName + ".fsm"
    val stats = ConfigCLI.resultsDir + "/" + patternName + ".stats.rec"
    val argsTest: Array[String] = Array(
      "recognition",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + ConfigCLI.testSet,
      "--domainSpecificStream:" + ConfigCLI.domainSpecificStream,
      "--streamArgs:" + ConfigCLI.gap,
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

  private def runTrainFMM(patternName: String): Unit = {
    val fsm = ConfigCLI.resultsDir + "/" + patternName + ".fsm"
    val mc = ConfigCLI.resultsDir + "/" + patternName + ".mc"
    val argsTrain: Array[String] = Array(
      "estimateMatrix",
      "--matrixEstimator:mle",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + ConfigCLI.trainSet,
      "--domainSpecificStream:" + ConfigCLI.domainSpecificStream,
      "--streamArgs:" + ConfigCLI.gap,
      "--outputMc:" + mc
    )
    WayebCLI.main(argsTrain)
  }

  private def runTestFMM(patternName: String): Unit = {
    val fsm = ConfigCLI.resultsDir + "/" + patternName + ".fsm"
    val stats = ConfigCLI.resultsDir + "/" + patternName + ".stats"
    val mc = ConfigCLI.resultsDir + "/" + patternName + ".mc"
    val argsTest: Array[String] = Array(
      "forecasting",
      "--threshold:" + ConfigCLI.threshold,
      "--maxSpread:" + ConfigCLI.maxSpread,
      "--horizon:" + ConfigCLI.horizon,
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--mc:" + mc,
      "--stream:" + ConfigCLI.testSet,
      "--domainSpecificStream:" + ConfigCLI.domainSpecificStream,
      "--streamArgs:" + ConfigCLI.gap,
      "--statsFile:" + stats,
      "--finalsEnabled:" + ConfigCLI.finalsEnabled.toString
    )
    WayebCLI.main(argsTest)
  }

}
