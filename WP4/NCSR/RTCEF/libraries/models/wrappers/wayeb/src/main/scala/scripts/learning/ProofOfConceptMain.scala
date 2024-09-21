package scripts.learning

import net.automatalib.words.Word
import net.automatalib.words.impl.Alphabets
import scala.io.Source

object ProofOfConceptMain extends App {

  val positiveExamples = readData("interesting_alive.txt")
  val negativeExamples = readData("not_interesting_alive.txt")

  val learner = RPNI(Alphabets.characters('a', 'h'))

  val result = learner.train(positiveExamples, negativeExamples)

  // -- Resulted DFA
  println(s"States: {${result.states.mkString(", ")}}")
  println(s"Transitions: {${result.transitions.mkString(", ")}}")
  println(s"Initial State: ${result.initialState}")
  println(s"Final States: {${result.finalStates.mkString(", ")}}")

  def readData(filename: String): Iterable[Word[Character]] = {

    val source = Source.fromFile(filename)
    val words = source.getLines.map { line =>
      Word.fromCharSequence(line)
    }.toList
    source.close

    words
  }
}
