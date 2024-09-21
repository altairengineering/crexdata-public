package scripts.data.text

import java.nio.file.{Files, Paths}

import com.github.tototoshi.csv.{CSVReader, CSVWriter}

object Txt2Stream {
  def Run(
      txtFilePath: String,
      streamFilePath: String,
      streamDirPath: String,
      foldsNo: Int
  ): Unit = {
    val writer = CSVWriter.open(streamFilePath)
    val bufferedSource = io.Source.fromFile(txtFilePath)
    var counter = 0
    for (line <- bufferedSource.getLines) {
      println(line)
      val chars = line.trim.toCharArray
      //println(chars.mkString)
      val spaced = chars.map(c => if (c.isLetter) c.toUpper else ' ')
      //println(spaced.mkString)
      val spaceSplit = spaced.mkString.split(" ").toList
      //println(spaceSplit)
      val removedSpaces = spaceSplit.filter(x => !x.equalsIgnoreCase(""))
      //println(removedSpaces)
      val mergedSingleSpace = removedSpaces.map(x => " " + x).mkString.trim
      println(mergedSingleSpace)
      val charsSingleSpace = mergedSingleSpace.toCharArray
      for (char <- charsSingleSpace) {
        counter += 1
        val row = if (char == ' ') List(counter.toString, "#") else List(counter.toString, char.toString)
        writer.writeRow(row)
      }
      if (charsSingleSpace.nonEmpty) {
        counter += 1
        val row = List(counter.toString, "#")
        writer.writeRow(row)
      }
    }
    bufferedSource.close
    writer.close()
    createFolds(streamFilePath, streamDirPath, foldsNo, counter)
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
