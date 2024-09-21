package engine

import com.typesafe.scalalogging.LazyLogging
import fsm.FSMInterface
import fsm.runtime.{MatchDump, Run}
import model.forecaster.ForecasterInterface
import model.forecaster.runtime.ForecasterRun
import profiler.WtProfiler
import stream.GenericEvent
import stream.array.EventStream
import utils.structures.CyclicBuffer
import workflow.provider.{FSMProvider, ForecasterProvider}

object FixedDistanceEngine {
  def apply(
             fsmp: FSMProvider,
             pp: ForecasterProvider,
             checkForEmitting: Boolean,
             finalsEnabled: Boolean
           ): FixedDistanceEngine = new FixedDistanceEngine(fsmp, pp, (-1, -1), checkForEmitting, finalsEnabled)

  def apply(
             fsmp: FSMProvider,
             pp: ForecasterProvider,
             distance: (Int, Int),
             checkForEmitting: Boolean,
             finalsEnabled: Boolean
           ): FixedDistanceEngine = new FixedDistanceEngine(fsmp, pp, distance, checkForEmitting, finalsEnabled)
}

class FixedDistanceEngine private (
                                    fsmp: FSMProvider,
                                    pp: ForecasterProvider,
                                    distance: (Int, Int),
                                    checkForEmitting: Boolean,
                                    finalsEnabled: Boolean
                                  ) extends LazyLogging {
  logger.info("Initializing engine...")
  // Assumes only a single FSM and a single Predictor are given. Plus no partitioning.
  private val fsm: FSMInterface = fsmp.provide().head
  private val predictor: ForecasterInterface = pp.provide().head
  private val run: Run = Run(fsm, checkForEmitting)
  private val md = new MatchDump()
  private val predictorRun = ForecasterRun(predictor, collectStats = true, finalsEnabled)
  run.register(md)
  run.register(predictorRun)
  logger.info("initialization complete.")

  def processStream(eventStream: EventStream): WtProfiler = {
    val taggedStream = if (distance._1 == -1) eventStream else tagEventsAsValidForPredictions(eventStream)
    require(eventStream.getSize == taggedStream.getSize)
    logger.info("Running engine on tagged stream")
    var matchesNo = 0
    var execTime: Long = 0
    val streamSize = taggedStream.getSize

    for (i <- 0 until streamSize) {
      val e = taggedStream.getEvent(i)
      val t1 = System.nanoTime()
      run.processEvent(e)
      val t2 = System.nanoTime()
      val detected = run.ceDetected
      if (detected) matchesNo += 1
      execTime += (t2 - t1)
    }
    val collector = predictorRun.getCollector
    val profiler = new WtProfiler()
    profiler.setGlobal(streamSize   = streamSize,
                       execTime     = execTime,
                       matchesNo    = matchesNo,
                       lockedRuns   = 1,
                       unlockedRuns = 0,
                       matchDump    = md
    )
    profiler.createEstimators(Map(0 -> List(collector)))
    profiler.estimateStats()
    logger.info("Done")
    profiler
  }

  private def tagEventsAsValidForPredictions(eventStream: EventStream): EventStream = {
    logger.info("Scanning stream to tag events that should emit forecasts")
    val buffer = CyclicBuffer[GenericEvent](distance._1)
    var detected = false
    val taggedStream = EventStream()
    val taggingRun = Run(fsm, checkForEmitting)
    for (i <- 0 until eventStream.getSize) {
      val e = eventStream.getEvent(i)
      taggingRun.processEvent(e)
      val nowDetected = taggingRun.ceDetected
      if (!detected & !nowDetected) {
        val (head, dropped) = buffer.pushElement(e)
        dropped match {
          case Some(x) => {
            val newEvent = GenericEvent(x.id, x.eventType, x.timestamp, Map[String, Boolean]("isEmitting" -> false))
            taggedStream.addEvent(newEvent)
          }
          case None => "?"
        }
      } else if (!detected & nowDetected) {
        //go back and tag
        val bufferedEvents = buffer.pop
        val oldestBound = bufferedEvents.size
        val howManyToTake = oldestBound - distance._2 + 1
        val eventsToBeTagged = bufferedEvents.takeRight(howManyToTake)
        val untaggedEvents = bufferedEvents.dropRight(howManyToTake)
        for (taggedEvent <- eventsToBeTagged.reverse) {
          val newEvent = GenericEvent(taggedEvent.id, taggedEvent.eventType, taggedEvent.timestamp, Map[String, Boolean]("isEmitting" -> true))
          taggedStream.addEvent(newEvent)
        }
        for (untaggedEvent <- untaggedEvents.reverse) {
          val newEvent = GenericEvent(untaggedEvent.id, untaggedEvent.eventType, untaggedEvent.timestamp, Map[String, Boolean]("isEmitting" -> false))
          taggedStream.addEvent(newEvent)
        }
        val complexEvent = GenericEvent(e.id, e.eventType, e.timestamp, Map[String, Boolean]("isEmitting" -> false))
        taggedStream.addEvent(complexEvent)
        detected = true
        buffer.clear()
      } else if (detected & !nowDetected) {
        buffer.pushElement(e)
        detected = false
      } else if (detected & nowDetected) {
        val newEvent = GenericEvent(e.id, e.eventType, e.timestamp, Map[String, Boolean]("isEmitting" -> false))
        taggedStream.addEvent(newEvent)
      }
    }
    val remaining = buffer.pop
    for (r <- remaining.reverse) {
      val newEvent = GenericEvent(r.id, r.eventType, r.timestamp, Map[String, Boolean]("isEmitting" -> false))
      taggedStream.addEvent(newEvent)
    }
    logger.info("Scanning complete")
    taggedStream
  }

}
