package ui.experiments.exposed2cli

import java.nio.file.{Files, Paths}

import com.github.tototoshi.csv.CSVWriter
import profiler.{ProfilerInterface, WtProfiler}

abstract class Stats(
                      resultsDirPathName: String,
                      patternName: String
                    ) {
  private val resultsDirPath = Paths.get(resultsDirPathName)
  if (!Files.exists(resultsDirPath) | !Files.isDirectory(resultsDirPath)) Files.createDirectory(resultsDirPath)
  private val resultsFileName = resultsDirPathName + "/" + patternName + ".csv"
  Files.deleteIfExists(Paths.get(resultsFileName))
  protected val writer: CSVWriter = CSVWriter.open(resultsFileName, append = true)

  protected var throughputs = List.empty[Double]
  protected var execTimes = List.empty[Double]
  protected var modelTimes = List.empty[Double]
  protected var wtTimes = List.empty[Double]
  protected var predTimes = List.empty[Double]
  protected var extraTimes = List.empty[Double]
  protected var sizes = List.empty[Double]
  protected var folds: Int = 0

  var header: List[String] = List.empty

  def reset(): Unit = {
    throughputs = List.empty
    execTimes = List.empty
    modelTimes = List.empty[Double]
    wtTimes = List.empty[Double]
    predTimes = List.empty[Double]
    extraTimes = List.empty[Double]
    sizes = List.empty[Double]
    folds = 0
  }

  def writeHeader(): Unit = {
    writer.writeRow(header)
  }

  def writeRow(row: List[String]): Unit = writer.writeRow(row)

  def close(): Unit = writer.close()

  def writeResultsRow(rowPrefix: List[String]): Unit

  def setModelTimes(times: List[Long]): Unit = {
    modelTimes = times.map(t => t.toDouble / 1000000)
  }

  def setWtTimes(times: List[Long]): Unit = {
    wtTimes = times.map(t => t.toDouble / 1000000)
  }

  def update(
              profiler: ProfilerInterface,
              automatonSize: Int,
              predTime: Long,
              extraTime: Long
            ): Unit

  def update(
              profiler: ProfilerInterface,
              automatonSize: Int
            ): Unit = update(profiler, automatonSize, 0, 0)

}
