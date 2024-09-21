package ui.experiments.exposed2cli

import com.typesafe.scalalogging.LazyLogging
import profiler.{ProfilerInterface, WtProfiler}
import breeze.stats.mean

class ClassificationStats(
                           resultsDirPathName: String,
                           patternName: String
                         ) extends Stats(resultsDirPathName, patternName) with LazyLogging {

  private var precisions = List.empty[Double]
  private var recalls = List.empty[Double]
  private var f1scores = List.empty[Double]
  private var specificities = List.empty[Double]
  private var accuracies = List.empty[Double]
  private var tps = List.empty[Double]
  private var tns = List.empty[Double]
  private var fps = List.empty[Double]
  private var fns = List.empty[Double]

  header = List(
    "order",
    "maxSpread",
    "minDistance",
    "maxDistance",
    "threshold",
    "maxNoStates",
    //
    "fold",
    //
    "avgPrecision",
    "avgRecall",
    "avgF1",
    "avgSpecificity",
    "avgAccuracy",
    "avgTP",
    "avgTN",
    "avgFP",
    "avgFN",
    "avgSize",
    "avgTrainTime",
    "avgThroughput",
    "extraTime",
    "modelTime",
    "wtTime",
    "predTime"
  )

  override def reset(): Unit = {
    super.reset()
    precisions = List.empty[Double]
    recalls = List.empty[Double]
    f1scores = List.empty[Double]
    specificities = List.empty[Double]
    accuracies = List.empty[Double]
    tps = List.empty[Double]
    tns = List.empty[Double]
    fps = List.empty[Double]
    fns = List.empty[Double]
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
      recall,
      f1,
      specificity,
      accuracy,
      tp, tn, fp, fn
      ) = (
      ((profiler.getStat("streamSize").toInt / profiler.getStat("execTime").toDouble) * 1000000000).toString,
      profiler.getStat("execTime"),
      profiler.getStatFor("precision", 0),
      profiler.getStatFor("recall", 0),
      profiler.getStatFor("f1", 0),
      profiler.getStatFor("specificity", 0),
      profiler.getStatFor("accuracy", 0),
      profiler.getStatFor("tp", 0),
      profiler.getStatFor("tn", 0),
      profiler.getStatFor("fp", 0),
      profiler.getStatFor("fn", 0)
    )
    sizes = automatonSize :: sizes
    throughputs = throughput.toDouble :: throughputs
    execTimes = (execTime.toDouble / 1000000) :: execTimes
    predTimes = (predTime.toDouble / 1000000) :: predTimes
    extraTimes = (extraTime.toDouble / 1000000) :: extraTimes
    precisions = precision.toDouble :: precisions
    recalls = recall.toDouble :: recalls
    f1scores = f1.toDouble :: f1scores
    specificities = specificity.toDouble :: specificities
    accuracies = accuracy.toDouble :: accuracies
    tps = tp.toDouble :: tps
    tns = tn.toDouble :: tns
    fps = fp.toDouble :: fps
    fns = fn.toDouble :: fns
    folds += 1
  }

  override def writeResultsRow(rowPrefix: List[String]): Unit = {
    (1 to folds).foreach(f => writeFoldResults(f, rowPrefix))
    writeAvgResults(rowPrefix)
  }

  private def writeAvgResults(rowPrefix: List[String]): Unit = {
    val avgSize = mean(sizes)
    val avgThroughput = mean(throughputs)
    val avgModelTime = mean(modelTimes)
    val avgWtTime = mean(wtTimes)
    val avgPredTime = mean(predTimes)
    val avgExtraTime = mean(extraTimes)
    val avgTrainTime = avgModelTime + avgWtTime + avgPredTime + avgExtraTime
    val validPrecisions = precisions.filter(p => p >= 0.0)
    val avgPrecision = if (validPrecisions.nonEmpty) mean(validPrecisions) else -1.0
    val avgRecall = mean(recalls)
    val validf1 = f1scores.filter(p => p >= 0.0)
    val avgF1 = if (validf1.nonEmpty) mean(validf1) else -1.0
    val avgSpecificity = mean(specificities)
    val avgAccuracy = mean(accuracies)
    val avgTP = mean(tps)
    val avgTN = mean(tns)
    val avgFP = mean(fps)
    val avgFN = mean(fns)
    val row = rowPrefix ::: List(
      "0",
      avgPrecision.toString,
      avgRecall.toString,
      avgF1.toString,
      avgSpecificity.toString,
      avgAccuracy.toString,
      avgTP.toString,
      avgTN.toString,
      avgFP.toString,
      avgFN.toString,
      avgSize.toString,
      avgTrainTime.toString,
      avgThroughput.toString,
      avgExtraTime.toDouble.toString,
      avgModelTime.toString,
      avgWtTime.toString,
      avgPredTime.toString
    )
    writer.writeRow(row)
  }

  private def writeFoldResults(
                                fold: Int,
                                rowPrefix: List[String]
                              ): Unit = {
    val row = rowPrefix ::: List(
      fold.toString,
      precisions(fold - 1).toString,
      recalls(fold - 1).toString,
      f1scores(fold - 1).toString,
      specificities(fold - 1).toString,
      accuracies(fold - 1).toString,
      tps(fold - 1).toString,
      tns(fold - 1).toString,
      fps(fold - 1).toString,
      fns(fold - 1).toString,
      sizes(fold - 1).toString,
      (modelTimes(fold - 1) + wtTimes(fold - 1) + predTimes(fold - 1) + extraTimes(fold - 1)).toString,
      throughputs(fold - 1).toString,
      extraTimes(fold - 1).toDouble.toString,
      modelTimes(fold - 1).toString,
      wtTimes(fold - 1).toString,
      predTimes(fold - 1).toString
    )
    writer.writeRow(row)
  }
}
