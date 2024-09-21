package scripts.learning

/**
  * Represents the structure of a DFA.
  *
  * @param states a set of states
  * @param transitions a set of state transitions
  * @param initialState an initial state
  * @param finalStates a set of final states
  */
case class DFAStructure(
    states: Set[Int],
    transitions: Set[(Int, Int, String)],
    initialState: Int,
    finalStates: Set[Int]
)
