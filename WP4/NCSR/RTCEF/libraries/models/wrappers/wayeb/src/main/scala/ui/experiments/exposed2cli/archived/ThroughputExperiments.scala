package ui.experiments.exposed2cli.archived

import com.github.tototoshi.csv.CSVWriter
import com.typesafe.scalalogging.LazyLogging
import model.waitingTime.ForecastMethod.ForecastMethod
import stream.StreamFactory
import ui.ConfigUtils
import workflow.provider.source.forecaster.ForecasterSourceSerialized
import workflow.provider.source.sdfa.SDFASourceSerialized
import workflow.provider.source.spsa.SPSASourceSerialized
import workflow.provider.source.spst.SPSTSourceSerialized
import workflow.provider._
import workflow.task.engineTask.ERFTask

object ThroughputExperiments extends LazyLogging {
  def RunExperiment(
      fsmType: String,
      patternName: String,
      domain: String,
      resultsDir: String,
      testFile: String,
      sdfaOrders: List[Int],
      horizon: Int,
      thresholds: List[Double],
      maxSpreads: List[Int],
      spreadMethod: ForecastMethod,
      distances: List[(Double, Double)],
      finalsEnabled: Boolean,
      totalIterations: Int,
      warmupIterations: Int
  ): Unit = {
    logger.info("Creating test stream source")
    val testStreamSource = StreamFactory.getDomainStreamSource(testFile, domain, List.empty)

    val resultsFileName = resultsDir + "/" + patternName + "Throughput.csv"
    val writer: CSVWriter = CSVWriter.open(resultsFileName, append = false)
    val header = List("order", "threshold", "maxSpread", "minDistance", "maxDistance", "throughput")
    writer.writeRow(header)
    var iteration = 0
    while (iteration < totalIterations) {

      logger.info("Sampling")
      val sample = sampleSetting(fsmType, sdfaOrders, thresholds, maxSpreads, distances)
      val (order, threshold, maxSpread, distance) = sample

      val fsmfn: String = resultsDir + "/" + fsmType + "_m" + order + ".fsm"
      logger.info("Creating FSM provider from " + fsmfn)
      val prov =
        fsmType match {
          case "sdfa" => SDFAProvider(SDFASourceSerialized(fsmfn))
          case "spsa" => SPSAProvider(SPSASourceSerialized(fsmfn))
          case "spst" => SPSTProvider(SPSTSourceSerialized(fsmfn))
          case "mean" => SDFAProvider(SDFASourceSerialized(fsmfn))
        }
      val fsmp = FSMProvider(prov)

      val wtifn: String = resultsDir + "/" + fsmType + "_WTI_m" + order + "_h" + horizon + "_t" + threshold + "_s" + maxSpread + "_m" + spreadMethod + ".wt"
      logger.info("Creating wti provider from " + wtifn)
      val pp = ForecasterProvider(ForecasterSourceSerialized(wtifn))

      logger.info("Running iteration " + iteration + " with " + sample)
      val erft = ERFTask(
        fsmp             = fsmp,
        pp               = pp,
        predictorEnabled = true,
        finalsEnabled    = finalsEnabled,
        expirationDeadline   = ConfigUtils.defaultExpiration,
        distance         = distance,
        streamSource     = testStreamSource,
        collectStats     = false,
        show = false
      )
      val profiler = erft.execute()
      if (iteration >= warmupIterations) {
        profiler.printProfileInfo()
        val throughput = ((profiler.getStat("streamSize").toInt / profiler.getStat("execTime").toDouble) * 1000000000).toString
        val row = List(
          order.toString,
          threshold.toString,
          maxSpread.toString,
          distance._1.toString,
          distance._2.toString,
          throughput
        )
        writer.writeRow(row)
      }
      iteration += 1
    }
    writer.close()
  }

  private def sampleSetting(
      fsmType: String,
      sdfaOrders: List[Int],
      thresholds: List[Double],
      maxSpreads: List[Int],
      distances: List[(Double, Double)]
  ): (Int, Double, Int, (Double, Double)) = {
    val order =
      if (fsmType.equalsIgnoreCase("mean")) 0
      else {
        val ro = scala.util.Random
        sdfaOrders(ro.nextInt(sdfaOrders.length))
      }

    val rt = scala.util.Random
    val threshold = thresholds(rt.nextInt(thresholds.length))

    val rm = scala.util.Random
    val maxSpread = maxSpreads(rm.nextInt(maxSpreads.length))

    val rd = scala.util.Random
    val distance = distances(rd.nextInt(distances.length))

    (order, threshold, maxSpread, distance)
  }

}
