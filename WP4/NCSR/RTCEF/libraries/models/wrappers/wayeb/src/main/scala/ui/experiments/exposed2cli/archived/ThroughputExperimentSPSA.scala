package ui.experiments.exposed2cli.archived

import com.github.tototoshi.csv.CSVWriter
import com.typesafe.scalalogging.LazyLogging
import model.waitingTime.ForecastMethod.ForecastMethod
import stream.StreamFactory
import ui.ConfigUtils
import workflow.provider.source.forecaster.ForecasterSourceSerialized
import workflow.provider.source.spsa.SPSASourceSerialized
import workflow.provider.{FSMProvider, ForecasterProvider, SPSAProvider}
import workflow.task.engineTask.ERFTask

object ThroughputExperimentSPSA extends LazyLogging {
  def RunExperiment(
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
      iterations: Int
  ): Unit = {

    logger.info("Creating test stream source")
    val testStreamSource = StreamFactory.getDomainStreamSource(testFile, domain, List.empty)
    val resultsFileName = resultsDir + "/" + patternName + "Throughput.csv"
    val writer: CSVWriter = CSVWriter.open(resultsFileName, append = true)
    val header = List("order", "threshold", "maxSpread", "minDistance", "maxDistance", "throughput")
    writer.writeRow(header)

    var iteration = 0
    while (iteration < iterations) {
      logger.info("Sampling")
      val sample = sampleSetting(sdfaOrders, thresholds, maxSpreads, distances)
      val (order, threshold, maxSpread, distance) = sample

      val spsaifn: String = resultsDir + "/SPSAI_m" + order + ".spsai"
      logger.info("Creating FSM provider from " + spsaifn)
      val spsap = SPSAProvider(SPSASourceSerialized(spsaifn))
      val fsmp = FSMProvider(spsap)

      val wtifn: String = resultsDir + "/SPSA_WTI_m" + order + "_h" + horizon + "_t" + threshold + "_s" + maxSpread + "_m" + spreadMethod + ".wti"
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
      if (iteration >= 10) {
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
      sdfaOrders: List[Int],
      thresholds: List[Double],
      maxSpreads: List[Int],
      distances: List[(Double, Double)]
  ): (Int, Double, Int, (Double, Double)) = {
    val ro = scala.util.Random
    val rt = scala.util.Random
    val rm = scala.util.Random
    val rd = scala.util.Random

    val order = sdfaOrders(ro.nextInt(sdfaOrders.length))
    val threshold = thresholds(rt.nextInt(thresholds.length))
    val maxSpread = maxSpreads(rm.nextInt(maxSpreads.length))
    val distance = distances(rd.nextInt(distances.length))

    (order, threshold, maxSpread, distance)
  }

}
