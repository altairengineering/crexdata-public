package scripts.data.cards

import java.nio.file.{Files, Paths}

/**
  * * Before running this, you need to set the WAYEB_HOME environment variable.
  */
object ProcessTrxs {
  def main(args: Array[String]): Unit = {
    val tmpPathName = Constants.dataDir + "/tmp"
    val tmpPath = Paths.get(tmpPathName)
    if (!Files.exists(tmpPath) | !Files.isDirectory(tmpPath)) Files.createDirectory(tmpPath)

    val p = Constants.patternFilePath

    val trimmedPathName: String = Constants.dataDir + "/trimmed"
    val trimmedPath = Paths.get(trimmedPathName)
    if (!Files.exists(trimmedPath) | !Files.isDirectory(trimmedPath)) Files.createDirectory(trimmedPath)
    val trimmedFile = trimmedPathName + "/" + Constants.numEvents + ".csv"

    println("Trimming")
    Trimmer.Run(Constants.generatedFile, trimmedFile, Constants.withReset)

    println("Finding matches")
    val matchesFilePath = tmpPathName + "/tmp_deleteme_" + p._3 + ".csv"
    MatchFinding.RunMatches(
      p._1,
      p._2,
      trimmedFile,
      matchesFilePath,
      tmpPathName
    )

    println("Enriching")
    val enrichedPathName: String = Constants.dataDir + "/enriched"
    val enrichedPath = Paths.get(enrichedPathName)
    if (!Files.exists(enrichedPath) | !Files.isDirectory(enrichedPath)) Files.createDirectory(enrichedPath)
    EnrichmentWithCEs.RunEnrichment(
      trimmedFile,
      matchesFilePath,
      enrichedPathName,
      p._3,
      Constants.foldsNo
    )
  }
}
