package scripts.data.synthetic

import java.nio.file.{Files, Paths}
import com.github.tototoshi.csv.{CSVReader, CSVWriter}

object EnrichmentWithCEs {

  private def writeEnriched(
      inCsvFilePath: String,
      topDirPathName: String,
      initialCEs: Map[String, List[Int]],
      foldsNo: Int
  ): Unit = {
    val thisDirPathName = topDirPathName + "/"
    val thisDirPath = Paths.get(thisDirPathName)
    val outCsvFilePathName = thisDirPath + "/allvideos.csv"
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
      val frame = line(0).toInt
      val idpair = line(1) + "-" + line(2)
      if (ces.contains(idpair)) {
        val pairCEs = ces(idpair)
        val nextCETimestamp = pairCEs.head
        if (frame < nextCETimestamp) {
          writer.writeRow(line ::: List(nextCETimestamp))
          linesNo += 1
        } else {
          val remainingCETimestamps = pairCEs.tail
          if (remainingCETimestamps.nonEmpty) {
            writer.writeRow(line ::: List(remainingCETimestamps.head))
            linesNo += 1
            ces += (idpair -> remainingCETimestamps)
          } else {
            ces -= idpair
            if (frame == nextCETimestamp) {
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

  }

}
