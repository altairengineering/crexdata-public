package workflow.task.engineTask

import com.typesafe.scalalogging.LazyLogging
import engine.LogLossEngine
import profiler.LogLossProfiler
import stream.array.EventStream
import workflow.provider.{FSMProvider, MarkovChainProvider}
import workflow.task.Task

object LogLossTask {
  /**
    * Constructor for LogLoss task.
    *
    * @param fsmp Provider for the FSM.
    * @param mcp Provider for the FSM's matrix.
    * @param eventStream The event stream.
    * @return A log-loss task.
    */
  def apply(
             fsmp: FSMProvider,
             mcp: MarkovChainProvider,
             eventStream: EventStream
           ): LogLossTask = new LogLossTask(fsmp, mcp, eventStream)
}

/**
  * Task for running estimation of average log-loss on a given stream/sequence:
  * l(P,X) = - (1/T)sum_{i=1,T}{log(P(x_{i}|x_{1}...x_{i-1}))}
  * See engine.LogLossEngine.
  *
  * @param fsmp Provider for the FSM.
  * @param mcp Provider for the FSM's matrix.
  * @param eventStream The event stream.
  */
class LogLossTask private (
                            fsmp: FSMProvider,
                            mcp: MarkovChainProvider,
                            eventStream: EventStream
                          ) extends Task with LazyLogging {

  private val engine = LogLossEngine(fsmp, mcp)

  /**
    * Processes stream and estimates log-loss.
    *
    * @return Profiler with average log-loss.
    */
  override def execute(): LogLossProfiler = {
    logger.info("Executing log-loss task...")
    val profiler = engine.processStream(eventStream)
    logger.info("Done with log-loss task")
    profiler
  }

}
