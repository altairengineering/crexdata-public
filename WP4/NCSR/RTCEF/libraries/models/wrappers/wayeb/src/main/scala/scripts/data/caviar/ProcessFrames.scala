package scripts.data.caviar

import java.io.File
import java.nio.file.{Files, Paths}

import utils.MiscUtils

object ProcessFrames {
  def main(args: Array[String]): Unit = {

    val tmpPathName = ConfigFrameProcessing.dataDir + "/tmp"
    val tmpPath = Paths.get(tmpPathName)
    if (!Files.exists(tmpPath) | !Files.isDirectory(tmpPath)) Files.createDirectory(tmpPath)

    val p = ConfigFrameProcessing.patternFilePath

    val d = new File(ConfigFrameProcessing.videoFilesDir)
    val fileNamesSorted =
      if (d.exists() & d.isDirectory) {
        d.list().toList.sorted.map(x => ConfigFrameProcessing.videoFilesDir + "/" + x)
      } else List.empty
    //val fileNames = MiscUtils.shuffleList(filenamesSorted)
    var filenamesRepeated: List[String] = List.empty
    for (i <- 1 to p._4) filenamesRepeated = filenamesRepeated ::: fileNamesSorted

    println("Creating paired stream from " + filenamesRepeated)
    PairMatching.CreatePairedStream(
      filenamesRepeated,
      ConfigFrameProcessing.pairedFile,
      ConfigFrameProcessing.sampling
    )

    println("Finding matches")

    val matchesFilePath = tmpPathName + "/tmp_deleteme_" + p._3 + ".csv"
    MatchFinding.RunMatches(
      p._1,
      p._2,
      ConfigFrameProcessing.pairedFile,
      matchesFilePath,
      tmpPathName
    )

    println("Enriching")
    val enrichedPathName: String = ConfigFrameProcessing.dataDir + "/enriched"
    val enrichedPath = Paths.get(enrichedPathName)
    if (!Files.exists(enrichedPath) | !Files.isDirectory(enrichedPath)) Files.createDirectory(enrichedPath)
    EnrichmentWithCEs.RunEnrichment(
      ConfigFrameProcessing.pairedFile,
      matchesFilePath,
      enrichedPathName,
      p._3,
      p._5
    )

  }

}
