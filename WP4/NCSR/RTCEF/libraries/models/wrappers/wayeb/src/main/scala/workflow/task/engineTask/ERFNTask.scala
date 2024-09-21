package workflow.task.engineTask

import com.typesafe.scalalogging.LazyLogging
import engine.ERFNEngine
import profiler.NextProfiler
import stream.source.{EmitMode, StreamSource}
import workflow.provider.{FSMProvider, ForecasterProvider}
import workflow.task.Task

object ERFNTask {
  /**
    * Constructor for ERFn task.
    *
    * @param fsmp Provider for the FSM.
    * @param pp Provider for the (next) predictor.
    * @param topk k.
    * @param streamSource The source for the event stream.
    * @return An ERFn task.
    */
  def apply(
             fsmp: FSMProvider,
             pp: ForecasterProvider,
             topk: Int,
             streamSource: StreamSource
           ): ERFNTask = new ERFNTask(fsmp, pp, topk, streamSource)
}

/**
  * Task for running a variant of forecasting where the goal is to forecast the k most probable patterns to be satisfied
  * after the immediately next event. See engine.ERFNEngine for more details.
  *
  * @param fsmp Provider for the FSM.
  * @param pp Provider for the (next) predictor.
  * @param topk k.
  * @param streamSource The source for the event stream.
  */
class ERFNTask private (
                         fsmp: FSMProvider,
                         pp: ForecasterProvider,
                         topk: Int,
                         streamSource: StreamSource
                       ) extends Task with LazyLogging {

  private val engine = ERFNEngine(fsmp, pp, topk)

  /**
    * Sends events to the engine.
    *
    * @return A profiler with statistics.
    */
  override def execute(): NextProfiler = {
    logger.info("Executing ERFN task...")
    val es = streamSource.emitEventsAndClose(EmitMode.BUFFER)
    val profiler = engine.processStream(es)
    logger.info("done.")
    profiler
  }

  def getEngine: ERFNEngine = engine

}
