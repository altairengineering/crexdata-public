package scripts.data.maritime

import com.github.tototoshi.csv.{CSVReader, CSVWriter}
import com.typesafe.scalalogging.LazyLogging
import java.nio.file.{Files, Paths}

object EnrichmentWithCEs extends LazyLogging {

  def RunEnrichment(
                     inCsvFilePath: String,
                     CEsFilePath: String,
                     enrichPath: String,
                     patternName: String,
                     foldsNo: Int
                   ): Unit = {

    val cesReader = CSVReader.open(CEsFilePath)
    val lines = cesReader.all()
    val groupedLines = lines.groupBy(l => l(1))
    val timestampsOnly = groupedLines.mapValues(l => l.map(x => x(0).toLong))
    val groupedSortedCEs = timestampsOnly.mapValues(l => l.sorted)
    cesReader.close()

    val patternPathName = enrichPath + "/" + patternName
    val patternPath = Paths.get(patternPathName)
    if (!Files.exists(patternPath) | !Files.isDirectory(patternPath)) Files.createDirectory(patternPath)
    println("Enriching with all vessels")
    val allCsvFilePath = writeAllVessels(inCsvFilePath, patternPathName, groupedSortedCEs, foldsNo)

    println("Enriching per vessel")
    writePerVessel(allCsvFilePath, patternPathName, groupedSortedCEs, foldsNo)
  }

  private def writePerVessel(
                              inCsvFilePath: String,
                              patternPathName: String,
                              initialCEs: Map[String, List[Long]],
                              foldsNo: Int
                            ): Unit = {
    println("Enriching for " + initialCEs.keySet)
    val writer = CSVWriter.open(patternPathName + "/vesselsWithCEs.csv")
    initialCEs.foreach(x => writer.writeRow(List(x._1, x._2.size.toString)))
    writer.close()
    initialCEs.foreach(x => writeForVessel(inCsvFilePath, patternPathName, x._1, foldsNo))
  }

  private def writeForVessel(
                              inCsvFilePath: String,
                              patternPathName: String,
                              cemmsi: String,
                              foldsNo: Int
                            ): Unit = {
    println("Enriching for " + cemmsi)
    val inputReader = CSVReader.open(inCsvFilePath)
    val it = inputReader.iterator
    val mmsiPathName = patternPathName + "/" + cemmsi
    val mmsiPath = Paths.get(mmsiPathName)
    if (!Files.exists(mmsiPath) | !Files.isDirectory(mmsiPath)) Files.createDirectory(mmsiPath)
    val outCsvFilePathVessel: String = mmsiPathName + "/" + cemmsi + ".csv"
    val writer = CSVWriter.open(outCsvFilePathVessel)
    var linesNo = 0
    while (it.hasNext) {
      val line = it.next().toList
      val mmsi = line(1)
      if (mmsi.equalsIgnoreCase(cemmsi)) {
        writer.writeRow(line)
        linesNo += 1
      }
    }
    inputReader.close()
    writer.close()
    createFolds(outCsvFilePathVessel, mmsiPathName, foldsNo, linesNo)
  }

  private def writeAllVessels(
                               inCsvFilePath: String,
                               topDirPathName: String,
                               initialCEs: Map[String, List[Long]],
                               foldsNo: Int
                             ): String = {
    var ces = initialCEs
    val inputReader = CSVReader.open(inCsvFilePath)
    val it = inputReader.iterator
    val thisDirPathName = topDirPathName + "/" + "all"
    val thisDirPath = Paths.get(thisDirPathName)
    val outCsvFilePathName = thisDirPath + "/allvessels.csv"
    val writer =
      if (Files.exists(thisDirPath) & Files.isDirectory(thisDirPath)) {
        CSVWriter.open(outCsvFilePathName)
      } else {
        Files.createDirectory(thisDirPath)
        CSVWriter.open(outCsvFilePathName)
      }
    var linesNo = 0
    while (it.hasNext) {
      val line = it.next().toList
      val mmsi = line(1)
      val timestamp = line(0).toLong
      if (ces.contains(mmsi)) {
        val mmsiCEs = ces(mmsi)
        val nextCETimestamp = mmsiCEs.head
        if (timestamp < nextCETimestamp) {
          writer.writeRow(line ::: List(nextCETimestamp))
          linesNo += 1
        } else {
          val remainingCETimestamps = mmsiCEs.tail
          if (remainingCETimestamps.nonEmpty) {
            writer.writeRow(line ::: List(remainingCETimestamps.head))
            linesNo += 1
            ces += (mmsi -> remainingCETimestamps)
          } else {
            ces -= mmsi
            if (timestamp == nextCETimestamp) {
              writer.writeRow(line ::: List(-1))
              linesNo += 1
            }
          }
        }
      }
    }
    inputReader.close()
    writer.close()
    createFolds(outCsvFilePathName, thisDirPathName, foldsNo, linesNo)
    outCsvFilePathName
  }

  private def createFolds(
                           inCsvFilePath: String,
                           inCsvDirPath: String,
                           foldsNo: Int,
                           linesNo: Int
                         ): Unit = {
    val testInterval = linesNo / foldsNo
    val foldLimits = for (f <- 0 until foldsNo) yield (1 + f * testInterval, 1 + (f + 1) * testInterval)
    val foldsDirPathName = inCsvDirPath + "/folds"
    val foldsDirPath = Paths.get(foldsDirPathName)
    if (!Files.exists(foldsDirPath) | !Files.isDirectory(foldsDirPath)) Files.createDirectory(foldsDirPath)
    val files = for (f <- 1 to foldsNo) yield (foldsDirPathName + "/fold" + f + "_train.csv", foldsDirPathName + "/fold" + f + "_test.csv")
    val trainWriters = files.map(f => CSVWriter.open(f._1))
    val testWriters = files.map(f => CSVWriter.open(f._2))
    val inputReader = CSVReader.open(inCsvFilePath)
    val it = inputReader.iterator
    var lineNo = 0
    while (it.hasNext) {
      val line = it.next()
      lineNo += 1
      for (fold <- 0 until foldsNo) {
        if (lineNo >= foldLimits(fold)._1 & lineNo < foldLimits(fold)._2) testWriters(fold).writeRow(line)
        else trainWriters(fold).writeRow(line)
      }
    }
    trainWriters.foreach(w => w.close())
    testWriters.foreach(w => w.close())
  }

}
