package ui.experiments.exposed2cli

import profiler.{ProfilerInterface, WtProfiler}
import breeze.stats.{mean, stddev}
import com.typesafe.scalalogging.LazyLogging

class RegressionStats(
                       resultsDirPathName: String,
                       patternName: String,
                       foldsNo: Int
                     ) extends Stats(resultsDirPathName, patternName) with LazyLogging {

  private var scores = List.empty[Double]
  private var precisions = List.empty[Double]
  private var losses = List.empty[Double]
  private var noForecastsRatios = List.empty[Double]
  private var actualSpreads = List.empty[Double]
  private var confidences = List.empty[Double]
  private var deviations = List.empty[Double]
  private var rmses = List.empty[Double]
  private var maes = List.empty[Double]
  private var mares = List.empty[Double]

  header = List(
    "order",
    "maxSpread",
    "minDistance",
    "maxDistance",
    "threshold",
    "maxNoStates",
    //
    "avgScore",
    //"stddevScore",
    "rmse",
    "mae",
    "mare",
    "avgPrecision",
    //"stddevPrecision",
    //"avgLoss",
    //"stddevLoss",
    "avgNoForecastsRatio",
    //"stddevNoForecastsRatio",
    "avgActualSpread",
    //"stddevActualSpread",
    "avgConfidence",
    //"stddevConfidence",
    "avgDeviation",
    //"stddevDeviation",
    "avgSize",
    //"stddevSize",
    "avgTrainTime",
    //"stddevTrainTime",
    "avgThroughput",
    //"stddevThroughput",
    "extraTime",
    "modelTime",
    "wtTime",
    "predTime"
  )

  override def reset(): Unit = {
    super.reset()
    scores = List.empty[Double]
    precisions = List.empty[Double]
    losses = List.empty[Double]
    noForecastsRatios = List.empty[Double]
    actualSpreads = List.empty[Double]
    confidences = List.empty[Double]
    deviations = List.empty[Double]
    rmses = List.empty[Double]
    maes = List.empty[Double]
    mares = List.empty[Double]
  }

  override def update(
                       profileri: ProfilerInterface,
                       automatonSize: Int,
                       predTime: Long,
                       extraTime: Long
                     ): Unit = {
    val profiler = profileri.asInstanceOf[WtProfiler]
    profiler.printProfileInfo()
    val (
      throughput,
      execTime,
      precision,
      actualSpread,
      intervalScore,
      predictionsNo,
      lossRatio,
      noForecastsRatio,
      dist,
      conf,
      dev,
      rmse,
      mae,
      mare
      ) = (
      ((profiler.getStat("streamSize").toInt / profiler.getStat("execTime").toDouble) * 1000000000).toString,
      profiler.getStat("execTime"),
      profiler.getStatFor("precision", 0),
      profiler.getStatFor("spread", 0),
      profiler.getStatFor("interval", 0),
      profiler.getStatFor("predictionsNo", 0),
      profiler.getStatFor("lossRatio", 0),
      profiler.getStatFor("noForecastsRatio", 0),
      profiler.getStatFor("distance", 0),
      profiler.getStatFor("confidence", 0),
      profiler.getStatFor("deviation", 0),
      profiler.getStatFor("rmse", 0),
      profiler.getStatFor("mae", 0),
      profiler.getStatFor("mare", 0)
    )
    sizes = automatonSize :: sizes
    throughputs = throughput.toDouble :: throughputs
    execTimes = (execTime.toDouble / 1000000) :: execTimes
    predTimes = (predTime.toDouble / 1000000) :: predTimes
    extraTimes = (extraTime.toDouble / 1000000) :: extraTimes
    losses = lossRatio.toDouble :: losses
    noForecastsRatios = noForecastsRatio.toDouble :: noForecastsRatios
    val thisScore = intervalScore.toDouble
    val thisPrecision = precision.toDouble
    val thisActualSpread = actualSpread.toDouble
    val thisConfidence = conf.toDouble
    val thisDeviation = dev.toDouble
    val thisRMSE = rmse.toDouble
    val thisMAE = mae.toDouble
    val thisMARE = mare.toDouble
    if (thisScore != -1.0) {
      scores = thisScore :: scores
      precisions = thisPrecision :: precisions
      actualSpreads = thisActualSpread :: actualSpreads
      confidences = thisConfidence :: confidences
      deviations = thisDeviation :: deviations
      rmses = thisRMSE :: rmses
      maes = thisMAE :: maes
      mares = thisMARE :: mares
    }
  }

  override def writeResultsRow(rowPrefix: List[String]): Unit = {
    val avgSize = mean(sizes)
    val stddevSize = stddev(sizes)
    val avgThroughput = mean(throughputs)
    //val stddevThroughput = stddev(throughputs)
    val avgTime = mean(execTimes)
    //val stddevTime = stddev(execTimes)
    val avgModelTime = mean(modelTimes)
    val avgWtTime = mean(wtTimes)
    val avgPredTime = mean(predTimes)
    val avgExtraTime = mean(extraTimes)
    val avgTrainTime = avgModelTime + avgWtTime + avgPredTime + avgExtraTime
    val (avgScore, stddevScore) = if (scores.nonEmpty) (mean(scores), stddev(scores)) else (-1.0, 0.0)
    val (avgPrecision, stddevPrecision) =
      if (precisions.nonEmpty) (mean(precisions), stddev(precisions)) else (-1.0, 0.0)
    val avgLoss = mean(losses)
    val stddevLoss = stddev(losses)
    val avgNoForecastsRatio = mean(noForecastsRatios)
    //val stddevNoForecastsRatio = stddev(noForecastsRatios)
    val (avgActualSpread, stddevActualSpread) =
      if (actualSpreads.nonEmpty) (mean(actualSpreads), stddev(actualSpreads)) else (-1.0, 0.0)
    val (avgConfidence, stddevConfidence) =
      if (confidences.nonEmpty) (mean(confidences), stddev(confidences)) else (-1.0, 0.0)
    val (avgDeviation, stddevDeviation) =
      if (deviations.nonEmpty) (mean(deviations), stddev(deviations)) else (-1.0, 0.0)
    val avgRMSE = if (rmses.nonEmpty) mean(rmses) else -1.0
    val avgMAE = if (maes.nonEmpty) mean(maes) else -1.0
    val avgMARE = if (mares.nonEmpty) mean(mares) else -1.0
    if (scores.nonEmpty & scores.size < foldsNo) logger.warn("Folds split between producing and non-producing")
    val row = rowPrefix ::: List(
      avgScore.toString,
      //stddevScore.toString,
      avgRMSE.toString,
      avgMAE.toString,
      avgMARE.toString,
      avgPrecision.toString,
      //stddevPrecision.toString,
      //avgLoss.toString,
      //stddevLoss.toString,
      avgNoForecastsRatio.toString,
      //stddevNoForecastsRatio.toString,
      avgActualSpread.toString,
      //stddevActualSpread.toString,
      avgConfidence.toString,
      //stddevConfidence.toString,
      avgDeviation.toString,
      //stddevDeviation.toString,
      avgSize.toString,
      //stddevSize.toString,
      avgTrainTime.toString,
      //stddevTrainTime.toString,
      avgThroughput.toString,
      //stddevThroughput.toString,
      avgExtraTime.toDouble.toString,
      avgModelTime.toString,
      avgWtTime.toString,
      avgPredTime.toString
    )
    writer.writeRow(row)
  }

}
