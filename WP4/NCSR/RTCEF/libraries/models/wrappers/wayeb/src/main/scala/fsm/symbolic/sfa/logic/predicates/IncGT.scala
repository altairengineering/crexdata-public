package fsm.symbolic.sfa.logic.predicates

import fsm.symbolic.sfa.logic.Predicate
import stream.GenericEvent

case class IncGT(arguments: List[String]) extends Predicate {

  override def evaluate(event: GenericEvent): Boolean = {
    val isFraud = event.getValueOf("isFraud").toString.toInt
    val fraudType = event.getValueOf("fraudType").toString.toInt
    isFraud == 1 & fraudType == 5
  }

  override def toString: String = "DecGT()"

}
