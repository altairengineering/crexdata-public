package scripts.data.synthetic

import com.github.tototoshi.csv.CSVWriter
import ui.WayebCLI

import scala.util.matching.Regex

object MatchFinding {

  def RunMatches(
      patternFilePath: String,
      datasetFilePath: String,
      matchesFilePath: String,
      tmpPathName: String
  ): Unit = {
    val patternName = "tmp_deleteme"
    runDis(patternName, patternFilePath, tmpPathName)
    runRec(patternName, datasetFilePath, tmpPathName)
    val matches = findMatches()
    val sorted = matches.sorted
    val writer: CSVWriter = CSVWriter.open(matchesFilePath)
    sorted.foreach(m => writer.writeRow(List(m.toString)))
    writer.close()
  }

  private def runDis(
      patternName: String,
      patternFile: String,
      tmpPathName: String
  ): Unit = {
    val fsm = tmpPathName + "/" + patternName + ".fsm"
    val argsTest: Array[String] = Array(
      "fsmDisambiguate",
      "--fsmType:symbolic",
      "--patterns:" + patternFile,
      "--declarations:" + "",
      "--outputFsm:" + fsm
    )
    WayebCLI.main(argsTest)
  }

  private def runRec(
      patternName: String,
      datasetFilename: String,
      tmpPathName: String
  ): Unit = {
    val fsm = tmpPathName + "/" + patternName + ".fsm"
    val stats = tmpPathName + "/" + patternName + ".stats.rec"
    val argsTest: Array[String] = Array(
      "recognition",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + datasetFilename,
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

  private def findMatches(): List[Long] = {
    val logPath = Config.wayebHome + "/wayeb.log"
    val mmsiPattern = new Regex("MATCH:Attr->([0-9]+)")
    val timePattern = new Regex("Timestamp->([0-9]+)")
    var matches: List[Long] = List.empty
    val bufferedSource = io.Source.fromFile(logPath)
    for (line <- bufferedSource.getLines) {
      var timestamp: Long = -1
      val timeMatch = timePattern.findFirstMatchIn(line)
      timeMatch match {
        case Some(x) => timestamp = x.group(1).toLong
        case _ => timestamp = -1
      }

      if (timestamp != -1) {
        matches = timestamp :: matches
      }

    }
    bufferedSource.close()
    matches.reverse
  }

}
