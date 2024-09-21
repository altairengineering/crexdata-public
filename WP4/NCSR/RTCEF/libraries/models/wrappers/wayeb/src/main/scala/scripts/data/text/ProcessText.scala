package scripts.data.text

import java.nio.file.{Files, Paths}

object ProcessText {

  def main(args: Array[String]): Unit = {

    val tmpPathName = Constants.outDir + "/tmp"
    val tmpPath = Paths.get(tmpPathName)
    if (!Files.exists(tmpPath) | !Files.isDirectory(tmpPath)) Files.createDirectory(tmpPath)

    val streamDir = Constants.outDir + "/stream"
    Txt2Stream.Run(Constants.txtFile, Constants.streamFile, streamDir, Constants.foldsNo)
  }

}
