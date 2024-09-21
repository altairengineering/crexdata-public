package engine

import com.typesafe.scalalogging.LazyLogging
import fsm.FSMInterface
import fsm.runtime.{Run, RunPool}
import model.markov.MarkovChain
import profiler.LogLossProfiler
import stream.GenericEvent
import stream.array.EventStream
import utils.MathUtils
import workflow.provider.{FSMProvider, MarkovChainProvider}
import breeze.stats.{mean, stddev}
import ui.ConfigUtils

object LogLossEngine {
  /**
    * Constructor for LogLoss engine.
    *
    * @param fsmProvider Provider for a finite state machine.
    * @param markovChainProvider Provider for a markov chain of engine.LogLossEngine#fsmProvider
    * @return A log-loss engine.
    */
  def apply(
             fsmProvider: FSMProvider,
             markovChainProvider: MarkovChainProvider
           ): LogLossEngine = new LogLossEngine(fsmProvider, markovChainProvider)
}

/**
  * Engine used only to calculate average log-loss on a given stream/sequence:
  * l(P,X) = - (1/T)sum_{i=1,T}{log(P(x_{i}|x_{1}...x_{i-1}))}
  *
  * Assumes only a single FSM and a single MC are given.
  * Used exclusively for testing. Not exposed to CLI.
  *
  * @param fsmProvider Provider for a finite state machine.
  * @param markovChainProvider Provider for a markov chain of engine.LogLossEngine#fsmProvider
  */
class LogLossEngine private (
                              fsmProvider: FSMProvider,
                              markovChainProvider: MarkovChainProvider
                            ) extends LazyLogging {
  logger.info("Initializing engine...")
  private val singlePartitionVal = ConfigUtils.singlePartitionVal
  private val fsm: FSMInterface = fsmProvider.provide().head
  private val mc: MarkovChain = markovChainProvider.provide().head
  private val runPool = RunPool(List(fsm), ConfigUtils.defaultExpiration, ConfigUtils.defaultDistance, show = false)
  //val run: Run = Run(fsm,(-1.0,1.0))
  logger.info("initialization complete.")

  /**
    * Processes streams and computes average log-loss.
    *
    * @param eventStream The input stream.
    * @return Statistics (avg log-loss) and sequence length/stream size.
    */
  def processStream(eventStream: EventStream): LogLossProfiler = {
    val streamSize = eventStream.getSize
    var numberOfEvents: Map[String, Int] = Map.empty
    var logSum: Map[String, Double] = Map.empty
    logger.info("Running Wayeb (log-loss) for " + streamSize + " events")
    for (i <- 0 until streamSize) {
      val e = eventStream.getEvent(i)
      val (currentState, nextState, isRunning, av) = processEvent(e)
      if (isRunning) {
        val transitionProb = mc.getTransProbFromTo(currentState, nextState)
        // The first transition of a SPSA may have "zero" probability.
        // The reason is that the initial start state of the SPSA is just a "guess".
        // We have to wait for the k first events of the stream before we can decide where the
        // start state will land. This first "transition" from the "guessed" to the actual first
        // state may have zero probability.
        // If more than one such warning is emitted (per partition), then something went wrong.
        if (transitionProb == 0.0) {
          logger.warn("ZERO TRANSITION PROB from/to " + currentState + "/" + nextState + " with " + e.toString)
        } else {
          val thisLog = MathUtils.logbase(transitionProb, 2)
          val thisSum = if (logSum.contains(av)) logSum(av) + thisLog else thisLog
          logSum += (av -> thisSum)
          val thisCounter = if (numberOfEvents.contains(av)) numberOfEvents(av) + 1 else 1
          numberOfEvents += (av -> thisCounter)
          //logger.info("Current log/length: " + logSum + "/" + T)
        }
      }
    }
    val losses: Map[String, Double] = logSum.map(x => (x._1, -(x._2 / numberOfEvents(x._1))))
    val avgLogLoss: Double = mean(losses.values)
    val stddevLogLoss: Double = stddev(losses.values)
    logger.info("Done. Average log-loss: " + avgLogLoss + "/" + stddevLogLoss)
    val llStats = LogLossProfiler(avgLogLoss, numberOfEvents.values.sum)
    llStats.estimateStats()
    llStats
  }

  /**
    * Processing of a single event.
    * If a SPSA is used as the FSM,
    * we might have to wait for some events until it actually starts running.
    *
    * @param event The event to be processed.
    * @return An Int for the state that the event led us and a Boolean to indicate whether the FSM
    *         is actually running or still waiting to find its first state.
    */
  private def processEvent(event: GenericEvent): (Int, Int, Boolean, String) = {
    val (run, av) = findRun(event)
    val currentState = run.getCurrentState
    run.processEvent(event)
    val nextState = run.getCurrentState
    (currentState, nextState, run.isRunning, av)
  }

  /**
    * Method that finds the appropriate run of an FSM that should do the actual processing of the event.
    *
    * @param event The event to be processed.
    * @return The appropriate run.
    */
  private def findRun(event: GenericEvent): (Run, String) = {
    val id = fsm.getId
    val partitionAttribute = fsm.getPartitionAttribute
    // First retrieve the value of the  partition attribute,
    // i.e., the attribute by which the input stream is to be partitioned.
    val av = if (partitionAttribute.equalsIgnoreCase(singlePartitionVal)) singlePartitionVal
    else event.getValueOf(partitionAttribute).toString
    // If the pool of runs already has a run with this partition value (for the ID of the FSM we
    // are examining), then get this run.
    if (runPool.existsRunWithAttributeVal(id, av)) {
      (runPool.getRunByAttribute(id, av), av)
    } // Otherwise, create a new run with this partition value.
    else {
      val r1 = runPool.checkOut(id, av, event.timestamp)
      (r1, av)
    }
  }

}
