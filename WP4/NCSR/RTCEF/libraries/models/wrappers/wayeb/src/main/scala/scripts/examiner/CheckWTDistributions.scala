package scripts.examiner

import com.github.tototoshi.csv.CSVWriter
import workflow.provider.source.matrix.MCSourceSerialized
import workflow.provider.source.sdfa.SDFASourceSerialized
import workflow.provider.{MarkovChainProvider, SDFAProvider}
import model.waitingTime.WtDistribution
import ui.ConfigUtils

import scala.collection.SortedMap

object CheckWTDistributions {
  def main(args: Array[String]): Unit = {
    val fsmFile = args(0)
    val mcFile = args(1)
    val fsmp = SDFAProvider(SDFASourceSerialized(fsmFile))
    val fsm = fsmp.provide().head
    val mp = MarkovChainProvider(MCSourceSerialized(mcFile))
    val mc = mp.provide().head

    val wtDistributions = mc.computeWTDists(fsm, ConfigUtils.defaultHorizon, ConfigUtils.defaultFinalsEnabled)
    val wtDistributionsSorted = wtDistributions.map(f => (f._1, f._2.getDistributionSorted))
    val wtString = SortedMap(wtDistributionsSorted.toSeq: _*).map(f => f._1.toString :: f._2.values.map(p => p.toString).toList).toList
    val wtSize = wtString.head.size
    val wtNo = wtString.size
    val outFile = args(2)
    val wr1 = CSVWriter.open(outFile + "_wts.csv")
    val headers1 = "timepoint" :: (0 until wtNo).map(s => "state_" + s.toString).toList
    val wtStringTranspose = (1 to wtSize - 1).map(t => getRow(wtString, t, List.empty)).toList
    wr1.writeRow(headers1)
    //wtString.foreach(w => wr1.writeRow(w))
    wtStringTranspose.foreach(w => wr1.writeRow(w))
    wr1.close()
    println(wtString)
    println(wtStringTranspose)

    val maxSpread = 60
    val thresholds = List(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    val preds = thresholds.map(t => buildPredictions(t, maxSpread, wtDistributions)).flatten

    val wr2 = CSVWriter.open(outFile + "_predictions.csv")
    val headers2 = List("threshold", "state", "start", "end", "prob")
    wr2.writeRow(headers2)
    preds.foreach(p => wr2.writeRow(p))
    wr2.close()

    println(preds)

  }

  def getRow(lin: List[List[String]], i: Int, lout: List[String]): List[String] = {
    lin match {
      case Nil => i.toString :: lout.reverse
      case head :: tail => getRow(tail, i, head(i) :: lout)
    }
  }

  def buildPredictions(
      threshold: Double,
      maxSpread: Int,
      wts: Map[Int, WtDistribution]
  ): List[List[String]] = {
    val preds = wts.mapValues(w => w.buildPrediction(threshold, maxSpread))
    val predsSorted = SortedMap(preds.toSeq: _*)
    val predsString = predsSorted.map(p => List(threshold.toString, p._1.toString, p._2.start.toString, p._2.end.toString, p._2.prob.toString)).toList
    predsString
  }

}
