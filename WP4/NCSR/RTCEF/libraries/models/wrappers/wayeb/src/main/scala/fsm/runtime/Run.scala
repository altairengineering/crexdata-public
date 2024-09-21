package fsm.runtime

import com.typesafe.scalalogging.LazyLogging
import db.DBConnector
import fsm.{FSMInterface, SPSAInterface, SPSTInterface}
import stream.{GenericEvent, ResetEvent}
import model.vmm.SymbolWord
import model.vmm.pst.CyclicBuffer
import ui.ConfigUtils

import scala.collection.mutable.ArrayBuffer

object Run {
  def apply(
             fsm: FSMInterface,
             defaultCheckForEmitting: Boolean,
             distance: (Double, Double),
             show: Boolean
           ): Run = new Run(fsm, defaultCheckForEmitting, distance, show)

  def apply(
             fsm: FSMInterface,
             defaultCheckForEmitting: Boolean
           ): Run =
    new Run(fsm, defaultCheckForEmitting, ConfigUtils.defaultDistance, ConfigUtils.defaultShowMatchesForecasts)

  def apply(
             fsm: FSMInterface,
             distance: (Double, Double),
             show: Boolean
           ): Run = new Run(fsm, false, distance, show)

  def apply(fsm: FSMInterface): Run =
    new Run(fsm, false, ConfigUtils.defaultDistance, ConfigUtils.defaultShowMatchesForecasts)
}

/**
  * Class representing an automaton run.
  *
  * @param fsm The automaton corresponding to the run.
  * @param checkForEmitting A special flag for determining how to emit forecasts. If true, the event must have an
  *                         "isEmitting" attribute and this determines whether we are allowed to emit a forecast or not.
  * @param distance A parameter (Tuple2, (minDistance,maxDistance)) that determines when forecasts should be emitted.
  *
  *                 If minDistance==maxDistance==-1, forecasts are emitted eagerly, whenever possible (i.e., whenever
  *                 the current state of a FSM can provide a valid forecast, forecast interval satisfying the confidence
  *                 threshold and max spread constraints).
  *
  *                 If minDistance!=-1 and minDistance < 1.0 (same for maxDistance), then forecasts are emitted only
  *                 from states whose remaining percentage is between minDistance and maxDistance.
  *                 The remaining percentage of a state is the ratio of its shortest path length to a final divided by
  *                 the maximum shortest path from all states. It gives an estimate of how close a state is to
  *                 completion. A remaining percentage of 0.0 means that the automaton is in a final state. A
  *                 remaining percentage of 1.0 means that it is in one of its start states.
  *
  *                 If minDistance!=-1 and minDistance >= 1.0 (same for maxDistance), forecasts are emitted only when
  *                 the current event's timestamp has a temporal distance from the next CE in the stream that is equal
  *                 to the value of the parameter.
  *                 CAUTION: This setting (with distances>=1.0) should be used only when one pattern is provided via
  *                 fsmProvider. Does not work for multiple patterns. Moreover, the input stream must have been
  *                 pre-processed so that each input event has an extra attribute, called "nextCETimestamp", providing
  *                 the timestamp of the next CE. See fsm.runtime.Run#isEmitting(stream.GenericEvent).
  * @param show Determines whether complex event matches and forecasts are to be displayed or not.
  */
class Run(
           fsm: FSMInterface,
           checkForEmitting: Boolean,
           distance: (Double, Double),
           show: Boolean
         ) extends RunPrototype with LazyLogging {
  private var currentState = fsm.getStartId
  private var detected = false
  private var listeners = ArrayBuffer[RunListener]()
  private val matchedEvents: Match = new Match()
  private var attributeValue = ""
  private var eventCounter: Long = 0
  // This buffer holds the last maxOrder + 1 events/symbols from the stream.
  // It is used by SPSA and SPST for determining when they are ready to start (when they have left the transient states)
  // and entered the recurrent class.
  // Also used by SPST to determine next state.
  private val buffer = fsm match {
    case x: SPSAInterface => new CyclicBuffer(x.maxOrder + 1)
    case x: SPSTInterface => new CyclicBuffer(x.maxOrder + 1)
    case _ => new CyclicBuffer(1)
  }
  // Flag that indicates whether the run has actually started checking for matches and emitting forecasts.
  // If the fsm is a SPSA or a SPST, we must first ensure that they are not in a transient state.
  // There is a subtle technical issue about the convertibility of a PST to a PSA.
  // Not every PST can be converted to a PSA (but every PST can be converted to a larger
  // class of so-call probabilistic automata). This is achievable under a certain condition. If this
  // condition does not hold, then the PST can be converted to an automaton that is composed
  // of a PSA as usual, with the addition of some extra states. These states, viewed as states
  // of a Markov chain, are transient. This means that the automaton will move through these
  // states for some transitions, but it will finally end into the states of the PSA, stay in that
  // class and never return to any of the transient states. In fact, if the automaton starts in any
  // of the transient states, then it will enter the single, recurrent class of the PSA in at most
  // mmax transitions. Given the fact that in our work we deal with streams of infinite length, it
  // is certain that, while reading a stream, the automaton will have entered the PSA after at
  // most mmax symbols. Thus, instead of checking this condition, we prefer to simply construct
  // only the PSA and wait (for at most mmax symbols) until the first m ≤ mmax symbols of a
  // stream have been consumed and are equal to a label of the PSA. At this point, we set the
  // current state of the PSA to the state with that label and start processing.
  private var started = fsm match {
    case _: SPSAInterface => false
    case _: SPSTInterface => false
    case _ => true
  }

  /**
    * Main method to process a new event, typically called from the engine.
    *
    * @param event The new event to be processed.
    * @return The time (in ns) it took the run to process the event (useful for profiling).
    */
  def processEvent(event: GenericEvent): Long = {
    event match {
      // Reset events are special events. It resets the run, returning it to its start state and clearing the stored
      // events.
      case _: ResetEvent => {
        currentState = fsm.getStartId
        detected = false
        matchedEvents.clear()
        buffer.clear()
        started = fsm match {
          case _: SPSAInterface => false
          case _: SPSTInterface => false
          case _ => true
        }
        send2Listeners(RunMessage.ResetMessage())
        0
      }
      case _ => {
        // If the fsm is a SPST, then we need to push the new event to our buffer so that we can later determine the
        // next state.
        if (fsm.isInstanceOf[SPSTInterface]) {
          val symbol = fsm.asInstanceOf[SPSTInterface].iso.evaluate(event)
          buffer.pushSymbol(symbol)
        }
        // Now determine whether the run can start running.
        val starting = isStarted(event)
        if (starting._1) {
          currentState = starting._2
          nowProcessEvent(event)
        } else 0
      }
    }
  }

  /**
    * Determines whether the run can start and, if yes, the state it is in.
    *
    * @param event The new event which will help us determine whether the run has started.
    * @return (true,state) if the run has started, (false, state) if not.
    */
  private def isStarted(event: GenericEvent): (Boolean, Int) = {
    // If the run has already started, we are ok. The run can keep running and we simply return the current state.
    if (started) (started, currentState)
    else {
      if (fsm.isInstanceOf[SPSAInterface]) {
        // If the FSM is a SPSA, we need to check.
        val startWith = canStart(event, fsm.asInstanceOf[SPSAInterface])
        startWith match {
          case Some(x) => {
            started = true
            (true, x)
          }
          case None => (false, currentState)
        }
      } else if (fsm.isInstanceOf[SPSTInterface]) {
        // Same if the FSM is a SPST.
        val startWith = canStart(fsm.asInstanceOf[SPSTInterface])
        startWith match {
          case Some(x) => {
            started = true
            (true, x)
          }
          case None => (false, currentState)
        }
      } else (true, currentState) // In every other case, we do not need to check. Just return true and the current state.
    }
  }

  /**
    * Determines whether the run can start in case of a SPSA.
    * This is the only place we push symbols to the buffer for SPSA. After a SPSA run has started, we no longer need to
    * mess with the buffer.
    *
    * @param event The new event.
    * @param spsai The SPSA interface.
    * @return The id of the state if the SPSA can start, otherwise None.
    */
  private def canStart(
                        event: GenericEvent,
                        spsai: SPSAInterface
                      ): Option[Int] = {
    val buffered = SymbolWord(buffer.pop)
    val startState = spsai.canStartWith(buffered)
    val symbol = spsai.iso.evaluate(event)
    buffer.pushSymbol(symbol)
    startState
  }

  /**
    * Determines whether the run can start in case of a SPST.
    *
    * @param spsti The SPST interface.
    * @return The id of the state if the SPST can start, otherwise None.
    */
  private def canStart(spsti: SPSTInterface): Option[Int] = {
    val buffered = SymbolWord(buffer.pop)
    val startState = spsti.canStartWith(buffered)
    startState
  }

  /**
    * The actual processing of the new event happens here.
    * We first find the next state, emit the match if it's a full match and notify listeners,
    * like predictor.runtime.PredictorRun and fsm.runtime.MatchDump.
    *
    * @param event The new event to process.
    * @return The time (in ns) it took to process the event.
    */
  private def nowProcessEvent(event: GenericEvent): Long = {
    val t1 = System.nanoTime()
    // Find the next state. For SPST, the buffer is also needed to determine the next state.
    val nextState = fsm.getNextState(currentState, event, buffer)
    // A counter that simply counts the events we process.
    // This is useful for determining absolute reference points for forecasts.
    // TODO: in a true streaming setting, you will probably get an overflow at some point.
    eventCounter += 1
    if (eventCounter == 0) {
      logger.warn("Event counter reached 0. Possible overflow.")
      throw new Error("Event counter reached 0. Possible overflow.")
    }
    detected = fsm.isFinal(nextState)
    //TODO: Events added even if they simply signify a virtual kill. Find a way to add only those that truly matter.
    matchedEvents.addEvent(event, eventCounter, nextState)
    // Determine whether we are allowed to emit a forecast.
    val emitting = isEmitting(event, nextState)
    val rm = RunMessage(
      fmDetected     = detected,
      currentState = nextState,
      previousState  = currentState,
      eventCounter = eventCounter,
      timestamp = event.timestamp.toLong,
      matchedEvents  = matchedEvents,
      attributeValue = attributeValue,
      isEmitting = emitting,
      lastEvent = event,
      show = show
    )
    if (detected) {
      matchedEvents.setFull(true)
      if (show)
        logger.info(
          "MATCH: " +
            "Attr->" + attributeValue +
            " Timestamp->" + event.timestamp.toLong +
            " State->" + nextState +
            " Events->" + matchedEvents.toString()
        )
      // We can write matches to a DB. CAUTION: this can really slow things down.
      if (ConfigUtils.write2db) DBConnector.writeNewDetection(rm)
    }
    send2Listeners(rm)
    currentState = nextState
    if (detected) {
      // Do not reset counter. We need it running continuously so that we can properly evaluate predictions from finals.
      //eventCounter = 0
      matchedEvents.clear()
    }
    val t2 = System.nanoTime()
    val td = t2 - t1
    td
  }

  /**
    * Determines whether we are allowed to emit a forecast.
    *
    * @param event The new event.
    * @param state The state we are in.
    * @return True if we are allowed to emit a forecast.
    */
  private def isEmitting(
                          event: GenericEvent,
                          state: Int
                        ): Boolean = {
    if (checkForEmitting) {
      // In this case, we check if the event has an attribute "isEmitting" and return this.
      event match {
        case _: ResetEvent => false
        case _ => if (event.hasAttribute("isEmitting")) event.getValueOf("isEmitting").toString.toBoolean else true
      }
    } else if (distance._1 != -1.0 & distance._1 < 1.0 & fsm.remainingPercentage.nonEmpty) {
      // if the distance is below 1.0, then we check the state
      fsm.remainingPercentage(state) >= distance._1 & fsm.remainingPercentage(state) <= distance._2
    } else if (distance._1 != -1.0 & event.hasAttribute("nextCETimestamp")) {
      // if the distance is above 1.0, the event must have an attribute "nextCETimestamp"
      val nextCETimestamp = event.getValueOf("nextCETimestamp").toString.toLong
      val timeDiff = nextCETimestamp - event.timestamp
      if (timeDiff >= distance._1.toLong & timeDiff <= distance._2.toLong) {
        event match {
          case _: ResetEvent => false
          case _ => true
        }
      } else false
    } else true
  }

  /**
    * Checks whether the run is running.
    * @return True if the fsm has started.
    */
  def isRunning: Boolean = started

  /**
    * Checks whether the run has detected a complex event.
    * @return True if a complex event has been detected.
    */
  def ceDetected: Boolean = detected

  /**
    * @return The run's current state.
    */
  def getCurrentState: Int = currentState

  /**
    * Registers a new run listener that will be listening to this run.
    * @param rl The new run listener.
    */
  def register(rl: RunListener): Unit = listeners += rl

  /**
    * Sends a message to all the listeners, so that they can also process the new event.
    * @param rm The run message.
    */
  def send2Listeners(rm: RunMessage): Unit = listeners.foreach(_.newEventProcessed(rm))

  /**
    * Shutting down.
    */
  def shutdown(): Unit = listeners.foreach(l => l.shutdown())

  /**
    * Sets the attribute value.
    * @param av The attribute value.
    */
  def setAttributeValue(av: String): Unit = attributeValue = av

  /**
    * @return The attribute value.
    */
  def getAttributeValue: String = attributeValue

  /**
    * Completely resets the run (including listeners and event counter).
    */
  def reset(): Unit = {
    currentState = fsm.getStartId
    detected = false
    listeners.clear()
    matchedEvents.clear()
    attributeValue = ""
    eventCounter = 0
  }

  /* Methods for the fsm.runtime.RunPrototype interface */

  /**
    * Creates a new run for the same FSM.
    *
    * @return A new run.
    */
  override def cloneRun(): Run = {
    // FSM should not be stateful! Otherwise, each run would need to have its own FSM.
    // Here, only ref to FSM passed to each new run. They share the same FSM.
    val newRun = new Run(fsm, checkForEmitting, distance, show)
    newRun
  }

  override def getId: Int = fsm.getId

}
