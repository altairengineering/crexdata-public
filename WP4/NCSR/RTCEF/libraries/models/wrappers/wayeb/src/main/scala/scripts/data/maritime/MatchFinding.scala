package scripts.data.maritime

import com.github.tototoshi.csv.CSVWriter
import ui.WayebCLI
import scala.util.matching.Regex

object MatchFinding {

  def RunMatches(
                  patternFilePath: String,
                  syncedSynopsesFilePath: String,
                  matchesFilePath: String,
                  minMatches: Int,
                  tmpPathName: String
                ): Unit = {
    val patternName = "tmp_deleteme"
    runDis(patternName, patternFilePath, tmpPathName)
    runRec(patternName, syncedSynopsesFilePath, tmpPathName)
    val matches = findMatches()
    val retained = matches.filter(m => m._2.size >= minMatches)
    val flattened = retained.map(m => (m._2.map(l => (m._1, l)))).flatten.toList
    val sorted = flattened.sortBy(_._2)
    val writer: CSVWriter = CSVWriter.open(matchesFilePath)
    sorted.foreach(m => writer.writeRow(List(m._2.toString, m._1.toString)))
    writer.close()
  }

  private def findMatches(): Map[Int, List[Long]] = {
    val logPath = Constants.wayebHome + "/wayeb.log"
    val mmsiPattern = new Regex("MATCH: Attr->([0-9]+)")
    val timePattern = new Regex("Timestamp->([0-9]+)")
    var matches: Map[Int, List[Long]] = Map.empty
    val bufferedSource = io.Source.fromFile(logPath)
    for (line <- bufferedSource.getLines) {
      //println("\nnew line  " + line)
      var mmsi: Int = -1
      var timestamp: Long = -1

      val mmsiMatch = mmsiPattern.findFirstMatchIn(line)
      mmsiMatch match {
        case Some(x) => mmsi = x.group(1).toInt
        case _ => mmsi = -1
      }

      val timeMatch = timePattern.findFirstMatchIn(line)
      timeMatch match {
        case Some(x) => timestamp = x.group(1).toLong
        case _ => timestamp = -1
      }

      if (mmsi != -1 & timestamp != -1) {
        if (matches.contains(mmsi)) {
          val theseMatches = matches(mmsi)
          val newMatches = timestamp :: theseMatches
          matches += (mmsi -> newMatches)
        } else matches += (mmsi -> List(timestamp))
      }

    }

    bufferedSource.close()
    matches
  }

  private def runDis(
                      patternName: String,
                      patternFile: String,
                      tmpPathName: String
                    ): Unit = {
    val fsm = tmpPathName + "/" + patternName + ".fsm"
    val argsTest: Array[String] = Array(
      "compile",
      "--patterns:" + patternFile,
      "--declarations:" + "",
      "--outputFsm:" + fsm
    )
    WayebCLI.main(argsTest)
  }

  private def runRec(
                      patternName: String,
                      syncedSynopsesFilePath: String,
                      tmpPathName: String
                    ): Unit = {
    val fsm = tmpPathName + "/" + patternName + ".fsm"
    val stats = tmpPathName + "/" + patternName + ".stats.rec"
    val argsTest: Array[String] = Array(
      "recognition",
      "--fsm:" + fsm,
      "--stream:" + syncedSynopsesFilePath,
      "--domainSpecificStream:" + Constants.domainSpecificStream,
      "--streamArgs:" + "",
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

}
