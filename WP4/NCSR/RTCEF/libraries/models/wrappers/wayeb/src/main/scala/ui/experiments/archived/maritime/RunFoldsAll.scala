package ui.experiments.archived.maritime

import com.github.tototoshi.csv.{CSVReader, CSVWriter}
import com.typesafe.scalalogging.LazyLogging
import ui.WayebCLI

object RunFoldsAll extends LazyLogging {
  val sampleInterval = 60
  val dataDir = System.getenv("IMIS_DATA") + "/" + sampleInterval + "sec"
  val predictionThresholds = List(0.1, 0.3, 0.5, 0.7, 0.9)
  val orders = List(1)
  val folds = 1
  val maxSpread: Int = (1 * 3600) / sampleInterval // 1 hour
  val horizon: Int = 1 * (24 * 3600) / sampleInterval // 1 day
  val home: String = System.getenv("WAYEB_HOME")
  val initialPatternName = "approachingPortsEurope"

  def main(args: Array[String]): Unit = {
    orders.foreach(m => runFoldsForPatternTrain(m))
    //orders.foreach(m => runFoldsForPatternTest(m))
    //orders.foreach(m => runFoldsForPatternRec(m))

  }

  private def runFoldsForPatternTrain(order: Int): Unit = {
    //val resultsPath = home + "/results/datacron/maritime/" + initialPatternName + "/m_" + order
    val resultsPath = "/mnt/Warehouse/experiments/results/datacron/maritime/execTimes/approachingPortsEurope"
    val patternName = initialPatternName + "_m_" + order
    logger.info("Running " + folds + " folds for training")
    (1 to folds).foreach(f => runFoldTrain(f, resultsPath, patternName))
  }

  private def runFoldsForPatternTest(order: Int): Unit = {
    //val resultsPath = home + "/results/datacron/maritime/" + initialPatternName + "/m_" + order
    val resultsPath = "/mnt/Warehouse/experiments/results/datacron/maritime/execTimes/approachingPortsEurope"
    val patternName = initialPatternName + "_m_" + order
    val gatheredStatsFile = resultsPath + "/" + patternName + ".stats_gathered.csv"
    logger.info("Running " + folds + " folds for testing")
    predictionThresholds.foreach(t => {
      logger.info("Running " + folds + " folds for threshold=" + t)
      (1 to folds).foreach(f => {
        runFoldTest(f, t, resultsPath, patternName)
      })
    })
    /*logger.info("Gathering stats...")
    val gathWr = CSVWriter.open(gatheredStatsFile,append = false)
    val row = List("threshold","avgExecTime","avgPrecision","avgSpread","avgDistance")
    gathWr.writeRow(row)
    gathWr.close()
    predictionThresholds.foreach(t => gatherStats(t,resultsPath,patternName,gatheredStatsFile))*/
  }

  private def runFoldsForPatternRec(order: Int): Unit = {
    //val resultsPath = home + "/results/datacron/maritime/" + initialPatternName + "/m_" + order
    val resultsPath = "/mnt/Warehouse/experiments/results/datacron/maritime/execTimes/approachingPortsEurope"
    val patternName = initialPatternName + "_m_" + order
    val gatheredStatsFile = resultsPath + "/" + patternName + ".stats_gathered.csv"
    logger.info("Running " + folds + " folds for testing")
    predictionThresholds.foreach(t => {
      logger.info("Running " + folds + " folds for threshold=" + t)
      (1 to folds).foreach(f => {
        runFoldRec(f, t, resultsPath, patternName)
      })
    })
    /*logger.info("Gathering stats...")
    val gathWr = CSVWriter.open(gatheredStatsFile,append = false)
    val row = List("threshold","avgExecTime","avgPrecision","avgSpread","avgDistance")
    gathWr.writeRow(row)
    gathWr.close()
    predictionThresholds.foreach(t => gatherStats(t,resultsPath,patternName,gatheredStatsFile))*/
  }

  private def gatherStats(
      threshold: Double,
      resultsPath: String,
      patternName: String,
      gatheredStatsFile: String
  ): Unit = {
    val recStatsFile = resultsPath + "/" + patternName + "_" + threshold + ".stats.csv"
    val recRd = CSVReader.open(recStatsFile)
    val recResults = recRd.all()
    recRd.close()
    val execTimes = recResults.map(r => r(1)).map(t => t.toLong)
    require(execTimes.size == folds)
    val avgExecTime = execTimes.sum / folds.toDouble

    val patternStatsFile = resultsPath + "/" + patternName + "_" + threshold + ".stats_pattern.csv"
    val patRd = CSVReader.open(patternStatsFile)
    val patResults = patRd.all()
    patRd.close()
    val precisions = patResults.map(r => r(2)).map(p => p.toDouble)
    require(precisions.size == folds)
    val spreads = patResults.map(r => r(6)).map(s => s.toDouble)
    require(spreads.size == folds)
    val distances = patResults.map(r => r(7)).map(d => d.toDouble)
    require(distances.size == folds)

    val validPrecisions = precisions.filter(p => p != -1.0)
    val avgPrecision = if (validPrecisions.isEmpty) {
      -1.0
    } else {
      if (validPrecisions.size != folds) {
        logger.warn("Some precisions @ -1.0")
      }
      validPrecisions.sum / validPrecisions.size.toDouble
    }

    val validSpreads = spreads.filter(s => s != -1.0)
    val avgSpread = if (validSpreads.isEmpty) {
      -1.0
    } else {
      if (validSpreads.size != folds) {
        logger.warn("Some spreads @ -1.0")
      }
      validSpreads.sum / validSpreads.size.toDouble
    }

    val validDistances = distances.filter(d => d != -1.0)
    val avgDistance = if (validDistances.isEmpty) {
      -1.0
    } else {
      if (validDistances.size != folds) {
        logger.warn("Some distances @ -1.0")
      }
      validDistances.sum / validDistances.size.toDouble
    }

    val gathWr = CSVWriter.open(gatheredStatsFile, append = true)
    val row = List(threshold.toString, avgExecTime.toString, avgPrecision.toString, avgSpread.toString, avgDistance.toString)
    gathWr.writeRow(row)
    gathWr.close()

  }

  private def runFoldTrain(
      foldNo: Int,
      resultsPath: String,
      patternName: String
  ): Unit = {
    val fsm = resultsPath + "/" + patternName + ".fsm"
    val trainSet = dataDir + "/train_" + foldNo + "_enriched.csv"
    val mc = resultsPath + "/" + patternName + "_fold_" + foldNo + ".mc"
    val argsTrain: Array[String] = Array(
      "estimateMatrix",
      "--percentTrain:0.99",
      "--matrixEstimator:mle",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + trainSet,
      "--domainSpecificStream:datacronMaritime",
      "--streamArgs:",
      "--outputMc:" + mc
    )
    WayebCLI.main(argsTrain)
  }

  private def runFoldTest(
      foldNo: Int,
      threshold: Double,
      resultsPath: String,
      patternName: String
  ): Unit = {
    val fsm = resultsPath + "/" + patternName + ".fsm"
    val testSet = dataDir + "/test_" + foldNo + "_enriched.csv"
    val stats = resultsPath + "/" + patternName + "_" + threshold + ".stats"
    val mc = resultsPath + "/" + patternName + "_fold_" + foldNo + ".mc"
    val argsTest: Array[String] = Array(
      "forecasting",
      "--threshold:" + threshold,
      "--maxSpread:" + maxSpread,
      "--horizon:" + horizon,
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--mc:" + mc,
      "--stream:" + testSet,
      "--domainSpecificStream:datacronMaritime",
      "--streamArgs:",
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

  private def runFoldRec(
      foldNo: Int,
      threshold: Double,
      resultsPath: String,
      patternName: String
  ): Unit = {
    val fsm = resultsPath + "/" + patternName + ".fsm"
    val testSet = dataDir + "/test_" + foldNo + "_enriched.csv"
    val stats = resultsPath + "/" + patternName + "_" + threshold + ".stats.rec"
    val argsTest: Array[String] = Array(
      "recognition",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + testSet,
      "--domainSpecificStream:datacronMaritime",
      "--streamArgs:",
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

  private def runFold(
      foldNo: Int,
      threshold: Double,
      resultsPath: String,
      patternName: String
  ): Unit = {
    val fsm = resultsPath + "/" + patternName + ".fsm"
    val trainSet = dataDir + "/train_" + foldNo + ".csv"
    val testSet = dataDir + "/test_" + foldNo + ".csv"
    val mc = resultsPath + "/" + patternName + ".mc"
    val stats = resultsPath + "/" + patternName + "_" + threshold + ".stats"

    val argsTrain: Array[String] = Array(
      "estimateMatrix",
      "--percentTrain:0.99",
      "--matrixEstimator:mle",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + trainSet,
      "--domainSpecificStream:datacronMaritime",
      "--streamArgs:",
      "--outputMc:" + mc
    )
    WayebCLI.main(argsTrain)

    val argsTest: Array[String] = Array(
      "forecasting",
      "--threshold:" + threshold,
      "--maxSpread:" + maxSpread,
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--mc:" + mc,
      "--stream:" + testSet,
      "--domainSpecificStream:datacronMaritime",
      "--streamArgs:",
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)

  }

}
