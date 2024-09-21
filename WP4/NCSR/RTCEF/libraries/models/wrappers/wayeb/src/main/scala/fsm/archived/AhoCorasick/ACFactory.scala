package fsm.archived.AhoCorasick

import fsm.CountPolicy._
import scala.collection.mutable.Set

class ACFactory {

}

object ACFactory {
  def buildACA(keywords: List[String], policy: CountPolicy, es: Set[Char]): ACAutomaton = {
    require(policy == NONOVERLAP | policy == OVERLAP)
    val s = keywords.size
    if (s < 1) throw new IllegalArgumentException("No keywords provided to construct AC automaton")

    if (s == 1) {
      val aca = new ACAutomatonSingle(keywords, policy, es)
      aca.buildACA()
    } else {
      val aca = new ACAutomatonMulti(keywords, policy, es)
      aca.buildACA()
    }
  }
}
