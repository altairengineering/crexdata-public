package scripts.learning

import de.learnlib.algorithms.rpni.BlueFringeRPNIDFA
import net.automatalib.automata.fsa.DFA
import net.automatalib.words.{Alphabet, Word}

import scala.collection.JavaConverters._

class RPNI[A] private (alphabet: Alphabet[A], learner: BlueFringeRPNIDFA[A]) {

  private def toModel[S](dfa: DFA[S, A]): DFAStructure = {
    val ids = dfa.stateIDs()

    var states = Set.empty[Int]
    var transitions = Set.empty[(Int, Int, String)]
    val initialState = ids.getStateId(dfa.getInitialState) + 1
    var finalStates = Set.empty[Int]

    dfa.getStates.asScala.foreach { state =>
      val stateID = ids.getStateId(state) + 1
      states += stateID
      if (dfa.isAccepting(state)) finalStates += stateID

      alphabet.asScala.foreach { symbol =>
        val nextState = dfa.getTransition(state, symbol)
        if (nextState != null)
          transitions += ((stateID, ids.getStateId(nextState) + 1, symbol.toString))
      }
    }

    DFAStructure(states, transitions, initialState, finalStates)
  }

  def train(positives: Iterable[Word[A]], negatives: Iterable[Word[A]]): DFAStructure = {
    learner.addPositiveSamples(positives.asJavaCollection)
    learner.addNegativeSamples(negatives.asJavaCollection)
    val automaton = learner.computeModel()
    toModel(automaton)
  }
}

object RPNI {

  def apply[A](alphabet: Alphabet[A]): RPNI[A] =
    new RPNI(alphabet, new BlueFringeRPNIDFA[A](alphabet))
}

