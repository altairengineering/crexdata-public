package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent
import utils.StringUtils.list2Str

case class GTE2D(args: List[String]) extends Predicate {

  val minDist = args(0).toDouble

  override def evaluate(event: GenericEvent): Boolean = {

    val x1 = event.getValueOf("x1").toString.toInt
    val y1 = event.getValueOf("y1").toString.toInt
    val x2 = event.getValueOf("x2").toString.toInt
    val y2 = event.getValueOf("y2").toString.toInt

    math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) >= minDist

  }

  override def toString: String = "GTE2D(" + list2Str(args, ",") + ")"

}
