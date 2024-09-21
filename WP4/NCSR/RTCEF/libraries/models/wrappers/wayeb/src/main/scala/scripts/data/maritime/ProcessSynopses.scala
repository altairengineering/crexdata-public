package scripts.data.maritime

import java.nio.file.{Files, Paths}

/**
  * Before running this, you need to set WAYEB_HOME and MARITIMEDIR environment variables.
  */
object ProcessSynopses {
  def main(args: Array[String]): Unit = {

    val tmpPathName = Constants.dataDir + "/tmp"
    val tmpPath = Paths.get(tmpPathName)
    if (!Files.exists(tmpPath) | !Files.isDirectory(tmpPath)) Files.createDirectory(tmpPath)

    // You can skip/comment out interpolation if you already have the interpolated data
    println("Interpolating")
    SyncedInterpolation.RunInterpolation(
      Constants.synopsesFilePath,
      Constants.intervalSec,
      Constants.maxGap,
      Constants.speedThreshold,
      Constants.startTime,
      Constants.endTime,
      Constants.syncedSynopsesFilePath
    )
    println("Interpolation finished")

    println("Finding matches")
    val p = Constants.patternFilePath
    val matchesFilePath = tmpPathName + "/tmp_deleteme_" + p._3 + ".csv"
    MatchFinding.RunMatches(p._1, Constants.syncedSynopsesFilePath, matchesFilePath, p._2, tmpPathName)

    println("Enriching")
    val enrichedPathName: String =
      Constants.dataDir +
        "/enriched/" +
        Constants.startTime + "-" + Constants.endTime +
        "_gap" + Constants.maxGap +
        "_interval" + Constants.intervalSec +
        "_speed" + Constants.speedThreshold +
        "_matches" + Constants.patternFilePath._2
    val enrichedPath = Paths.get(enrichedPathName)
    if (!Files.exists(enrichedPath) | !Files.isDirectory(enrichedPath)) Files.createDirectory(enrichedPath)
    EnrichmentWithCEs.RunEnrichment(
      Constants.syncedSynopsesFilePath,
      matchesFilePath,
      enrichedPathName,
      p._3,
      Constants.foldsNo
    )
    println("Enriching done")

  }
}
