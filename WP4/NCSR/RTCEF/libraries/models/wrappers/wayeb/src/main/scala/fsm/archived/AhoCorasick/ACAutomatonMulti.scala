package fsm.archived.AhoCorasick

import scala.collection.mutable.Set
import fsm.CountPolicy._

private[AhoCorasick] class ACAutomatonMulti(k: List[String], p: CountPolicy, es: Set[Char]) extends ACAutomatonI {
  require(k.size > 1)
  private val keywords = k
  private val policy = p
  private val extraSymbols = es

  def buildACA(): ACAutomaton = {
    val aca = new ACAutomaton(keywords, extraSymbols)
    aca.setCountPolicy(policy)
    aca.build()
    aca
  }

}
