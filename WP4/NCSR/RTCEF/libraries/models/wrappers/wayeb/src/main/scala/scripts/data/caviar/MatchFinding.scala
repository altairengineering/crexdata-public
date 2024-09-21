package scripts.data.caviar

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
    sorted.foreach(m => writer.writeRow(List(m._1.toString, m._2.toString, m._3.toString)))
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
      "fsmDisambiguate",
      "--fsmType:symbolic",
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
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + datasetFilename,
      "--domainSpecificStream:caviar",
      "--streamArgs:",
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

  private def findMatches(): Set[(Int, Int, Int)] = {
    val logPath = ConfigFrameProcessing.wayebHome + "/wayeb.log"
    val idpairPattern = new Regex("MATCH:Attr->([0-9]+-[0-9]+)")
    val framePattern = new Regex("Timestamp->([0-9]+)")
    var matches: Set[(Int, Int, Int)] = Set.empty
    val bufferedSource = io.Source.fromFile(logPath)
    for (line <- bufferedSource.getLines) {
      var idpair: (Int, Int) = (-1, -1)
      var frame: Int = -1

      val idpairMatch = idpairPattern.findFirstMatchIn(line)
      idpairMatch match {
        case Some(x) => idpair = {
          val idstr = x.group(1).split("-")
          (idstr(0).toInt, idstr(1).toInt)
        }
        case _ => idpair = (-1, -1)
      }

      val frameMatch = framePattern.findFirstMatchIn(line)
      frameMatch match {
        case Some(x) => frame = x.group(1).toInt
        case _ => frame = -1
      }

      if (idpair._1 != -1 & frame != -1) {
        val newMatch = (frame, idpair._1, idpair._2)
        matches += newMatch
      }
    }
    bufferedSource.close()
    matches
  }

}
