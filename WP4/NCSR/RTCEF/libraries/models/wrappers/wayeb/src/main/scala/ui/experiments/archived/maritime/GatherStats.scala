package ui.experiments.archived.maritime

import com.github.tototoshi.csv.{CSVReader, CSVWriter}
import com.typesafe.scalalogging.LazyLogging

object GatherStats extends LazyLogging {
  val featuresList = List("all")
  val orders = List(1)
  val folds = 10
  val predictionThresholds = List(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  val home: String = System.getenv("WAYEB_HOME")
  val initialPatternName = "approachingBrestPort"

  def main(args: Array[String]): Unit = {
    featuresList.foreach(f => {
      orders.foreach(m => {
        val resultsPath = home + "/results/datacron/maritime/" + initialPatternName + "/features_" + f + "_m_" + m
        val patternName = initialPatternName + "_features_" + f + "_m_" + m
        val gatheredStatsFile = resultsPath + "/" + patternName + ".stats_gathered.csv"
        val gathWr = CSVWriter.open(gatheredStatsFile, append = false)
        val row = List("threshold", "avgExecTime", "avgPrecision", "avgSpread", "avgDistance")
        gathWr.writeRow(row)
        gathWr.close()
        predictionThresholds.foreach(t => gatherStats(t, resultsPath, patternName, gatheredStatsFile))
      })
    })
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

}
