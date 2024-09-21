package scripts.learning

import com.typesafe.scalalogging.LazyLogging
import stream.StreamFactory
import model.waitingTime.ForecastMethod
import ui.ConfigUtils
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.{FSMProvider, ForecasterProvider, SDFAProvider, SPSTProvider, WtProvider}
import workflow.provider.source.sdfa.{SDFASourceDirectI, SDFASourceLLDFA}
import workflow.provider.source.spst.{SPSTSourceDirectI, SPSTSourceFromSDFA}
import workflow.provider.source.wt.WtSourceSPST
import workflow.task.engineTask.ERFTask

object ProofOfConceptMain1_Classification extends LazyLogging {

  def main(args: Array[String]): Unit = {

    // manually constructing a simple DFA, recognizes instances of a

    // two states, with a, we move from state 1 (start) to state 2 (final)
    // also added the other transitions to make it deterministic
    val states: Set[Int] = Set(1, 2)
    val symbols: Set[String] = Set("b", "c", "d", "e", "f", "g", "h")
    // with a, go to state 2
    val transition1To2WithA: Set[(Int, Int, String)] = Set((1, 2, "a"))
    // with everything else, stay in state 1
    val loop1Transitions: Set[(Int, Int, String)] = symbols.map(symbol => (1, 1, symbol)).toSet
    // when in state 2, with a stay here
    val loop2Transition: Set[(Int, Int, String)] = Set((2, 2, "a"))
    // with anything else, go back to state 1
    val transitions2To1: Set[(Int, Int, String)] = symbols.map(symbol => (2, 1, symbol)).toSet
    val transitions = transition1To2WithA ++ loop1Transitions ++ loop2Transition ++ transitions2To1
    val start = 1
    val finals: Set[Int] = Set(2)
    // now we have everything we need

    // Also set the order. This is the order of the Markov chain to be constructed later. Not part of the DFA. Just a
    // parameter.
    // CAUTION: high values of the order may lead to an explosion in the number of states.
    // If you set the order to 0, you will see that precision falls below 1.0.
    val order = 1

    // we can change the input arguments and just pass a DFAStructure object
    val sdfap0 = SDFAProvider(SDFASourceLLDFA(states, transitions, start, finals, order = 0, streaming = false))
    // provide() typically returns a list of SDFAs. Here, we only have one SDFA, so just get the first one.
    val sdfa = sdfap0.provide().head
    //logger.info("SDFA: \n" + sdfa.toString)
    val sdfap = SDFAProvider(SDFASourceDirectI(List(sdfa)))

    // now running training and forecasting

    // don't forget to set WAYEB_HOME
    val home = System.getenv("WAYEB_HOME")
    // i have created a dummy file for testing
    val simulationsFile = home + "/scripts/data/bio/simulations.csv"

    // create the source for the stream of input data
    val streamSource = StreamFactory.getCSVStreamSource(simulationsFile)

    // first, create a provider for the SPST (symbolic probabilistic suffix tree)
    val spstp0 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamSource, pMin = 0.001, alpha = 0.0, gamma = 0.001, r = 1.05))
    val spst = spstp0.provide().head
    val spstp = SPSTProvider(SPSTSourceDirectI(List(spst)))

    // create a FSM provider, just a wrapper around the SPST provider
    val fsmp = FSMProvider(spstp)

    // second, create a provider for the waiting-time distributions (you need the SPST provider for this)
    val distance = (0.0001, 1.0)
    val wtp = WtProvider(WtSourceSPST(spstp, horizon = 100, cutoffThreshold = 0.01, distance = distance))

    // third, create a provider for the forecast intervals (you need the distributions provider for this)
    val pp = ForecasterProvider(ForecasterSourceBuild(fsmp, wtp, horizon = 100, confidenceThreshold = 0.5, maxSpread = 4, method = ForecastMethod.CLASSIFY_NEXTK))

    val erft = ERFTask(
      fsmp             = fsmp,
      pp               = pp,
      predictorEnabled = true,
      finalsEnabled    = false,
      expirationDeadline   = ConfigUtils.defaultExpiration,
      distance         = distance,
      streamSource     = streamSource
    )
    val prof = erft.execute()
    prof.printProfileInfo()

  }

}
