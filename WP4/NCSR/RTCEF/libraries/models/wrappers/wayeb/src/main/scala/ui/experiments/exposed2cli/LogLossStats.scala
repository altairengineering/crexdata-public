package ui.experiments.exposed2cli

import com.typesafe.scalalogging.LazyLogging
import profiler.{LogLossProfiler, ProfilerInterface}
import breeze.stats.mean

class LogLossStats(
                    resultsDirPathName: String,
                    patternName: String
                  ) extends Stats(resultsDirPathName, patternName) with LazyLogging {

  private var losses = List.empty[Double]

  header = List(
    "order",
    "maxNoStates",
    //
    "avgLogLoss",
    "avgSize"
  )

  override def reset(): Unit = {
    super.reset()
    losses = List.empty[Double]
  }

  override def update(
                       profileri: ProfilerInterface,
                       automatonSize: Int,
                       predTime: Long,
                       extraTime: Long
                     ): Unit = {
    val profiler = profileri.asInstanceOf[LogLossProfiler]
    profiler.printProfileInfo()
    val logLoss = profiler.avgLogLoss
    losses = logLoss :: losses
    sizes = automatonSize :: sizes
  }

  override def writeResultsRow(rowPrefix: List[String]): Unit = {
    val avgSize = mean(sizes)
    val avgLoss = mean(losses)
    val row = rowPrefix ::: List(
      avgLoss.toString,
      avgSize.toString
    )
    writer.writeRow(row)
  }

}
