package scripts.data.cards

import java.nio.file.{Files, Paths}
import com.github.tototoshi.csv.{CSVReader, CSVWriter}

object Folder {

  def createFolds(
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
