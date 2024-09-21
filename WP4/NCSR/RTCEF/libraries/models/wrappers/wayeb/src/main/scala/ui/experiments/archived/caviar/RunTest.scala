package ui.experiments.archived.caviar

import com.github.tototoshi.csv.CSVReader
import scripts.data.caviar.ConfigFrameProcessing
import ui.WayebCLI

import scala.util.matching.Regex

object RunTest {
  val home: String = System.getenv("WAYEB_HOME")
  val dataDir: String = System.getenv("CAVIARDIR")
  val resultsDir: String = home + "/results/caviar"
  val trainDatasetFilename: String = dataDir + "/enriched/meetingSituation/folds/fold1_train.csv"
  val testDatasetFilename: String = dataDir + "/enriched/meetingSituation/folds/fold1_test.csv"
  val ce2check: Set[String] = Set("meeting", "fighting")
  val patternFile: String = home + "/patterns/caviar/meeting/situation.sre"
  val declarationsFile: String = home + "/patterns/caviar/meeting/declarationsSituation.sre"
  val policy: String = "nonoverlap"
  val finalsEnabled: String = "false"
  val threshold = 0.5
  val maxSpread = 10
  val horizon = 200

  def main(args: Array[String]): Unit = {
    runDis("test")
    runRec("test")
    //estimateF1()
    runTrain("test")
    runTest("test")
  }

  private def runDis(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val argsTest: Array[String] = Array(
      "fsmDisambiguate",
      "--fsmType:symbolic",
      "--patterns:" + patternFile,
      "--declarations:" + declarationsFile,
      "--countPolicy:" + policy,
      "--outputFsm:" + fsm
    )
    WayebCLI.main(argsTest)
  }

  private def runRec(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val stats = resultsDir + "/" + patternName + ".stats.rec"
    val argsTest: Array[String] = Array(
      "recognition",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + trainDatasetFilename,
      "--domainSpecificStream:caviar",
      "--streamArgs:",
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }

  private def estimateF1(): (Double, Double, Double) = {
    val matches = findMatches()
    val groundTruth = findGroundTruth()
    val truePositives = matches & groundTruth
    val falsePositives = matches &~ groundTruth
    val falseNegatives = groundTruth &~ matches
    val precision = truePositives.size.toDouble / (truePositives.size + falsePositives.size)
    val recall = truePositives.size.toDouble / (truePositives.size + falseNegatives.size)
    val f1 = (2 * precision * recall) / (precision + recall)
    println("Precision/Recall/F1 : " + precision + "/" + recall + "/" + f1)
    println("False Positives: " + falsePositives)
    (precision, recall, f1)
  }

  private def findGroundTruth(): Set[(Int, Int, Int)] = {
    var groundTruth: Set[(Int, Int, Int)] = Set.empty
    val reader = CSVReader.open(trainDatasetFilename)
    val it = reader.iterator
    while (it.hasNext) {
      val line = it.next()
      val frame = line(0).toInt
      if (frame != -1) {
        val object1Str = line(1).split(";")
        val object2Str = line(2).split(";")
        val groupStr = line(3).split(";")
        if (groupStr.length > 1) {
          val gcon = groupStr(2)
          if (ce2check.contains(gcon)) {
            val id1 = object1Str(0).toInt
            val id2 = object2Str(0).toInt
            val idpair = (id1, id2)
            val newMatch = (frame, id1, id2)
            groundTruth += newMatch
          }
        }
      }
    }
    reader.close()
    groundTruth
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

  private def runTrain(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val mc = resultsDir + "/" + patternName + ".mc"
    val argsTrain: Array[String] = Array(
      "estimateMatrix",
      "--matrixEstimator:mle",
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--stream:" + trainDatasetFilename,
      "--domainSpecificStream:caviar",
      "--streamArgs:",
      "--outputMc:" + mc
    )
    WayebCLI.main(argsTrain)
  }

  private def runTest(patternName: String): Unit = {
    val fsm = resultsDir + "/" + patternName + ".fsm"
    val stats = resultsDir + "/" + patternName + ".stats"
    val mc = resultsDir + "/" + patternName + ".mc"
    val argsTest: Array[String] = Array(
      "forecasting",
      "--threshold:" + threshold,
      "--maxSpread:" + maxSpread,
      "--horizon:" + horizon,
      "--fsmType:symbolic",
      "--fsm:" + fsm,
      "--mc:" + mc,
      "--stream:" + testDatasetFilename,
      "--domainSpecificStream:caviar",
      "--streamArgs:",
      "--finalsEnabled:" + finalsEnabled,
      "--statsFile:" + stats
    )
    WayebCLI.main(argsTest)
  }
}
