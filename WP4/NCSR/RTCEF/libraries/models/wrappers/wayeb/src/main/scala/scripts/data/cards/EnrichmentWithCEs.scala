package scripts.data.cards

import java.nio.file.{Files, Paths}

import com.github.tototoshi.csv.{CSVReader, CSVWriter}

object EnrichmentWithCEs {
  def RunEnrichment(
                     inCsvFilePath: String,
                     CEsFilePath: String,
                     enrichPath: String,
                     patternName: String,
                     foldsNo: Int
                   ): Unit = {
    val cesReader = CSVReader.open(CEsFilePath)
    val lines = cesReader.all()
    val groupedLines = lines.groupBy(l => l(0))
    val timestampsOnly = groupedLines.mapValues(l => l.map(x => x(1).toInt))
    val groupedSortedCEs = timestampsOnly.mapValues(l => l.sorted)
    cesReader.close()

    val patternPathName = enrichPath + "/" + patternName
    val patternPath = Paths.get(patternPathName)
    if (!Files.exists(patternPath) | !Files.isDirectory(patternPath)) Files.createDirectory(patternPath)
    println("Enriching")
    writeEnriched(inCsvFilePath, patternPathName, groupedSortedCEs, foldsNo)
  }

  private def writeEnriched(
                             inCsvFilePath: String,
                             topDirPathName: String,
                             initialCEs: Map[String, List[Int]],
                             foldsNo: Int
                           ): Unit = {
    val thisDirPathName = topDirPathName + "/"
    val thisDirPath = Paths.get(thisDirPathName)
    val outCsvFilePathName = thisDirPath + "/all.csv"
    val writer =
      if (Files.exists(thisDirPath) & Files.isDirectory(thisDirPath)) {
        CSVWriter.open(outCsvFilePathName)
      } else {
        Files.createDirectory(thisDirPath)
        CSVWriter.open(outCsvFilePathName)
      }
    var linesNo = 0
    var ces = initialCEs
    val inputReader = CSVReader.open(inCsvFilePath)
    val it = inputReader.iterator
    while (it.hasNext) {
      val line = it.next().toList
      val trxNo = line(0).toInt
      val pan = line(2)
      if (ces.contains(pan)) {
        val panCEs = ces(pan)
        val nextCETimestamp = panCEs.head
        if (trxNo < nextCETimestamp) {
          writer.writeRow(line ::: List(nextCETimestamp))
          linesNo += 1
        } else {
          val remainingCETimestamps = panCEs.tail
          if (remainingCETimestamps.nonEmpty) {
            writer.writeRow(line ::: List(remainingCETimestamps.head))
            linesNo += 1
            ces += (pan -> remainingCETimestamps)
          } else {
            ces -= pan
            if (trxNo == nextCETimestamp) {
              writer.writeRow(line ::: List(-1))
              linesNo += 1
            }
          }
        }
      } else {
        writer.writeRow(line ::: List(-1))
        linesNo += 1
      }
    }
    inputReader.close()
    writer.close()
    createFolds(outCsvFilePathName, thisDirPathName, foldsNo, linesNo)
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
