package engine

import com.typesafe.scalalogging.LazyLogging
import fsm.runtime.{Run, RunPool}
import model.forecaster.runtime.{ForecasterRun, ForecasterRunFactory}
import profiler.NextProfiler
import stream.array.EventStream
import ui.ConfigUtils
import utils.Progressor
import workflow.provider.{FSMProvider, ForecasterProvider}

object ERFNEngine {
  /**
    * Constructor for ERFn engine.
    *
    * @param fsmProvider The provider for the FSM(s) that run recognition.
    * @param predProvider The provider for the predictor(s) that emit forecasts.
    * @param topk The value for the parameter k.
    * @return An ERFn engine.
    */
  def apply(
             fsmProvider: FSMProvider,
             predProvider: ForecasterProvider,
             topk: Int
           ): ERFNEngine = new ERFNEngine(fsmProvider, predProvider, topk)
}

/**
  * This is a variant of the engine for recognition and forecasting.
  * The setting is the following: We have n variants of a pattern, provided as n different
  * patterns in the same file. After every new event arrival,
  * we know that at least one CE will be detected, i.e., one of the patterns will be satisfied.
  * For example, in text processing, we have an alphabet of 26 symbols (ignoring white spaces).
  * We can then create 26 patterns, one for detecting each symbol.
  * The goal is to forecast the k most probable patterns to be satisfied after the immediately next event.
  * For example, find the k most probable characters to appear next in a text.
  * To be used mostly for testing and evaluation.
  * Only works for one pattern (its n variants).
  * Partition attributes ignored. Always single partition used.
  *
  * @param fsmProvider The provider for the FSM(s) that run recognition.
  * @param predProvider The provider for the predictor(s) that emit forecasts.
  * @param topk The value for the parameter k.
  */
class ERFNEngine private (
                           fsmProvider: FSMProvider,
                           predProvider: ForecasterProvider,
                           topk: Int
                         ) extends LazyLogging {
  logger.info("Initializing engine...")
  require(topk > 0)
  private val fsmList = fsmProvider.provide()
  private val predList = predProvider.provide()
  private val runs = initializeRuns
  require(topk <= fsmList.size)
  logger.info("Initialization complete.")

  /**
    * Method to process the stream.
    * The stream is provided as an ArrayBuffer, since we need to know its size.
    *
    *
    * @param eventStream The input stream.
    * @return A profiler containing statistics.
    */
  def processStream(eventStream: EventStream): NextProfiler = {
    val streamSize = eventStream.getSize
    var execTime: Long = 0
    val profiler = NextProfiler(streamSize)
    logger.info("Running Wayeb (next) for " + streamSize + " events")
    val progressor = Progressor("ERFNEngine", streamSize, 5)
    for (i <- 0 until streamSize) {
      val e = eventStream.getEvent(i)
      val t1 = System.nanoTime()
      runs.foreach(r => r._1.processEvent(e))
      val matches = runs.map(r => r._1.ceDetected)
      val lastPredictions = getTopK
      //val lastPredictions = getTopKDummy(matches)
      val t2 = System.nanoTime()
      execTime += (t2 - t1)
      logger.info("Top-K: " + lastPredictions)
      // Do not add the first detection, since we have no forecasts for it.
      if (i > 0) profiler.addNewDetection(matches)
      // Do not add forecasts for after last event, since we will never see the next event.
      if (i < (streamSize - 1)) profiler.addNewForecast(lastPredictions)
      progressor.tick
    }
    profiler.setExecTime(execTime)
    profiler.estimateStats()
    logger.info("done.")
    profiler
  }

  /**
    * Retrieves the k most confident forecasts out of the n pattern FSMs.
    *
    * @return The k most confident forecasts, along with their confidence.
    */
  private def getTopK: List[(Int, Double)] = {
    // First emit forecasts from all runs.
    val predictions = runs.map(r => (r._2.getId, r._2.getCurrentPrediction))
    // Then sort them by their confidence and keep the k most confident.
    val topKpredictions = predictions.sortWith { case ((_, x), (_, y)) => x.prob > y.prob }.take(topk)
    //predictions.sortBy(_._2.prob).reverse.take(topk)
    topKpredictions.map(p => (p._1, p._2.prob))
  }

  /**
    * Creates the n runs and the corresponding predictors.
    *
    * @return The n runs and their predictors.
    */
  private def initializeRuns: List[(Run, ForecasterRun)] = {
    val fsmIDs = fsmList.map(fsm => fsm.getId)
    val runPool = RunPool(fsmList, ConfigUtils.defaultExpiration, ConfigUtils.defaultDistance, ConfigUtils.defaultShowMatchesForecasts)
    val fsmRuns = fsmIDs.map(fsmid => runPool.checkOut(fsmid, ConfigUtils.singlePartitionVal, 1))
    val predFactory = ForecasterRunFactory(predList, collectStats = true, finalsEnabled = true)
    val predRuns = fsmIDs.map(fsmid => predFactory.getNewForecasterRun(fsmid))
    val runs = fsmRuns.zip(predRuns)
    runs.foreach(r => r._1.register(r._2))
    runs
  }

  /**
    * A variant of getTopK selecting k "dummy" forecasts. m of them are the FSMs that detected
    * a CE in the previous step/event. This is a kind of "inertia". We forecast that if FSM i
    * detected a match, it will keep detecting it. The rest k-m forecasts are randomly chosen.
    *
    * @param matches A list of Booleans indicating which FSMs have detected a match.
    *                If the ith element is true, then the ith FSM has detected a match.
    * @return The k dummy forecasts. All confidence values are the same, 1.0/k.
    */
  private def getTopKDummy(matches: List[Boolean]): List[(Int, Double)] = {
    // First find the m FSMs (their indices) that have detected a match.
    val trueIndices = matches.zipWithIndex.filter(_._1 == true).map(e => e._2)
    var indices = trueIndices
    // For the rest k - m forecasts, randomly pick a FSM.
    for (i <- 1 to topk - trueIndices.size) {
      indices = sample(indices.toSet, matches.size) :: indices
    }
    val pr = List.fill(indices.size)(1.0 / indices.size)
    indices.zip(pr)
  }

  /**
    * Randomly picks an Int, up to max. Ints in excluded cannot be picked.
    *
    * @param excluded Set of Ints to be excluded from picking.
    * @param max The max Int that can be picked.
    * @return A random Int, up to max, that does not belong to excluded.
    */
  private def sample(
                      excluded: Set[Int],
                      max: Int
                    ): Int = {
    val seed = 100
    val rand = new scala.util.Random(seed)
    var sample = rand.nextInt(max + 1)
    while (excluded.contains(sample)) sample = rand.nextInt(max + 1)
    sample
  }

}
