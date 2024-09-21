package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class Close(args: List[String]) extends Predicate {
  private val minDist = args(0).toDouble.toInt
  private val maxDist = args(1).toDouble.toInt

  override def evaluate(event: GenericEvent): Boolean = {
    val distance = event.getValueOf("dist").toString.toInt
    distance >= minDist & distance < maxDist
  }

  override def toString: String = "Close(" + list2Str(args, ",") + ")"

}
