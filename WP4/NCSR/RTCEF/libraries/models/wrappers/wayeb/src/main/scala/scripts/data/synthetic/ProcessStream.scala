package scripts.data.synthetic

import java.nio.file.{Files, Paths}

object ProcessStream {
  def main(args: Array[String]): Unit = {
    val tmpPathName = Config.psaDir + "/tmp"
    val tmpPath = Paths.get(tmpPathName)
    if (!Files.exists(tmpPath) | !Files.isDirectory(tmpPath)) Files.createDirectory(tmpPath)

    PSAStreamGenerator.generate()
    val matchesFilePath = tmpPathName + "/tmp_deleteme_matches.csv"

    //MatchFinding.RunMatches(Config.patternFile,Config.trainFile,matchesFilePath,tmpPathName)
  }

}
