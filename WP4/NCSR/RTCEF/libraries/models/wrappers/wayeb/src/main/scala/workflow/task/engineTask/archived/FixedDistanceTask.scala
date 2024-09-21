package workflow.task.engineTask.archived

import com.typesafe.scalalogging.LazyLogging
import engine.FixedDistanceEngine
import profiler.WtProfiler
import stream.array.EventStream
import workflow.provider.{FSMProvider, ForecasterProvider}
import workflow.task.Task

object FixedDistanceTask {
  def apply(
             fsmp: FSMProvider,
             pp: ForecasterProvider,
             checkForEmitting: Boolean,
             finalsEnabled: Boolean,
             eventStream: EventStream
           ): FixedDistanceTask = new FixedDistanceTask(fsmp, pp, (-1, -1), checkForEmitting, finalsEnabled, eventStream)

}

class FixedDistanceTask private (
                                  fsmp: FSMProvider,
                                  pp: ForecasterProvider,
                                  distance: (Int, Int),
                                  checkForEmitting: Boolean,
                                  finalsEnabled: Boolean,
                                  eventStream: EventStream
                                ) extends Task with LazyLogging {

  private val engine = FixedDistanceEngine(fsmp, pp, distance, checkForEmitting, finalsEnabled)

  override def execute(): WtProfiler = {
    logger.info("Executing fixed distance task...")
    val profiler = engine.processStream(eventStream)
    logger.info("Done with fixed distance task")
    profiler
  }

}
