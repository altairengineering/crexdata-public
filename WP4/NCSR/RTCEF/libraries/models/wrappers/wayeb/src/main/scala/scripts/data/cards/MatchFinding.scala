package scripts.data.cards

import com.github.tototoshi.csv.CSVWriter
import ui.WayebCLI
import scala.util.matching.Regex

object MatchFinding {
  def RunMatches(
                  patternFilePath: String,
                  policy: String,
                  datasetFilePath: String,
                  matchesFilePath: String,
                  tmpPathName: String
                ): Unit = {
    val patternName = "tmp_deleteme"
    runDis(patternName, patternFilePath, tmpPathName, policy)
    runRec(patternName, datasetFilePath, tmpPathName)
    val matches = findMatches()
    val sorted = matches.toList.sortBy(_._1)
    val writer: CSVWriter = CSVWriter.open(matchesFilePath)
    sorted.foreach(m => writer.writeRow(List(m._1, m._2.toString)))
    writer.close()
  }

  private def runDis(
                      patternName: String,
                      patternFile: String,
                      tmpPathName: String,
                      policy: String
                    ): Unit = {
    val fsm = tmpPathName + "/" + patternName + ".fsm"
    val argsTest: Array[String] = Array(
      "compile",
      "--patterns:" + patternFile,
      "--declarations:" + "",
      "--countPolicy:" + policy,
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
      "--fsm:" + fsm,
      "--stream:" + datasetFilename,
      "--domainSpecificStream:cards",
      "--streamArgs:",
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

  private def findMatches(): Set[(String, Int)] = {
    val logPath = Constants.wayebHome + "/wayeb.log"
    val panPattern = new Regex("Attr->([0-9|a-z]+)")
    val trxPattern = new Regex("Timestamp->([0-9]+)")
    var matches: Set[(String, Int)] = Set.empty
    val bufferedSource = io.Source.fromFile(logPath)
    for (line <- bufferedSource.getLines) {
      var trx: Int = -1
      var pan: String = ""

      val panMatch = panPattern.findFirstMatchIn(line)
      panMatch match {
        case Some(x) => pan = x.group(1)
        case _ => pan = ""
      }

      val trxMatch = trxPattern.findFirstMatchIn(line)
      trxMatch match {
        case Some(x) => trx = x.group(1).toInt
        case _ => trx = -1
      }

      if (trx != -1) {
        val newMatch = (pan, trx)
        matches += newMatch
      }
    }
    bufferedSource.close()
    matches
  }

}
