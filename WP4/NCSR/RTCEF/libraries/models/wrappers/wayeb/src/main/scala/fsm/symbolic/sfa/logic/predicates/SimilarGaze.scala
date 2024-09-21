package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent

case class SimilarGaze(args: List[String]) extends Predicate {

  override def evaluate(event: GenericEvent): Boolean = {

    val gaze1 = event.getValueOf("gaze1").toString.toInt
    val gaze2 = event.getValueOf("gaze2").toString.toInt

    if (gaze1 == -1 || gaze2 == -1)
      return true
    else if (math.abs(gaze1 - gaze2) <= 45)
      return true
    return false

  }

}
