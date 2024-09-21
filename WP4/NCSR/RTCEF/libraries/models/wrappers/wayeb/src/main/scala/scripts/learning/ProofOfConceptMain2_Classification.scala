package scripts.learning

import java.io.{File, PrintStream}

import com.typesafe.scalalogging.LazyLogging
import net.automatalib.words.Word
import net.automatalib.words.impl.Alphabets
import stream.StreamFactory
import model.waitingTime.ForecastMethod
import ui.ConfigUtils
import workflow.provider.source.matrix.MCSourceMLE
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.{FSMProvider, ForecasterProvider, MarkovChainProvider, SDFAProvider, SPSTProvider, WtProvider}
import workflow.provider.source.sdfa.{SDFASourceDirectI, SDFASourceLLDFA}
import workflow.provider.source.spst.{SPSTSourceDirectI, SPSTSourceFromSDFA}
import workflow.provider.source.wt.{WtSourceMatrix, WtSourceSPST}
import workflow.task.engineTask.ERFTask

import scala.collection.JavaConverters._
import scala.io.Source

object ProofOfConceptMain2_Classification extends App with LazyLogging {

  final val folds = 5
  final val order = 3
  final val alphabet = Alphabets.characters('a', 'h')
  final val pathToPositiveExamples = "scripts/data/bio/interesting_alive.txt"
  final val pathToNegativeExamples = "scripts/data/bio/not_interesting_alive.txt"
  // if true, runs experiments with new method using variable-order models (VMM)
  // if false, runs experiments with old method, using full-order models (FMM)
  // Training with FMMs (especially with high orders) can lead to inconsistent matrices
  final val vmm: Boolean = true

  val positiveExamples = readData(pathToPositiveExamples)
  val negativeExamples = readData(pathToNegativeExamples)

  var k = 1
  makeStratifiedSets(positiveExamples, negativeExamples, folds).foreach {
    case (trainPosSet, trainNegSet, testSet) =>

      println(s"Fold $k:")

      val learner = RPNI(alphabet)
      val result = learner.train(trainPosSet, trainNegSet)

      // -- Resulted DFA
      logger.info(s"States: {${result.states.mkString(", ")}}")
      logger.info(s"Transitions: {${result.transitions.mkString(", ")}}")
      logger.info(s"Initial State: ${result.initialState}")
      logger.info(s"Final States: {${result.finalStates.mkString(", ")}}")

      val trainFile = fromStringsToCSVStream(trainPosSet ++ trainNegSet, "train")
      //val streamTrainSource = StreamFactory.getCSVStreamSource(trainFile.getAbsolutePath)
      val streamTrainSource = StreamFactory.getDomainStreamSource(trainFile.getAbsolutePath, domain = "bio", List.empty)
      val testFile = fromStringsToCSVStream(testSet, "test")
      //val streamTestSource = StreamFactory.getCSVStreamSource(testFile.getAbsolutePath)
      val streamTestSource = StreamFactory.getDomainStreamSource(testFile.getAbsolutePath, domain = "bio", List.empty)

      // minimum distance just above 0.0 to avoid producing forecasts from final states (finals have a distance of 0.0)
      val distance = (0.0001, 1.0)

      if (vmm) {
        // Create the learned automaton
        val sdfap0 = SDFAProvider(SDFASourceLLDFA(
          result.states, result.transitions, result.initialState, result.finalStates, order = 0, streaming = false
        ))
        val sdfa = sdfap0.provide().head
        val sdfap = SDFAProvider(SDFASourceDirectI(List(sdfa)))
        val spstp0 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource, pMin = 0.001, alpha = 0.0, gamma = 0.001, r = 1.05))
        val spst = spstp0.provide().head
        val spstp = SPSTProvider(SPSTSourceDirectI(List(spst)))
        val fsmp = FSMProvider(spstp)
        val wtp = WtProvider(WtSourceSPST(spstp, horizon = 10, cutoffThreshold = 0.01, distance = distance))
        val pp = ForecasterProvider(ForecasterSourceBuild(fsmp, wtp, horizon = 10, confidenceThreshold = 0.5, maxSpread = 4, method = ForecastMethod.CLASSIFY_NEXTK))

        val erft = ERFTask(
          fsmp             = fsmp,
          pp               = pp,
          predictorEnabled = true,
          finalsEnabled    = false,
          expirationDeadline   = ConfigUtils.defaultExpiration,
          distance         = distance,
          streamSource     = streamTestSource
        )
        val prof = erft.execute()
        prof.printProfileInfo()
      } else {
        val sdfap0 = SDFAProvider(SDFASourceLLDFA(
          result.states, result.transitions, result.initialState, result.finalStates, order = order, streaming = false, disambiguate = true
        ))
        val sdfa = sdfap0.provide().head
        val sdfap = SDFAProvider(SDFASourceDirectI(List(sdfa)))
        val fsmp = FSMProvider(sdfap)
        val mp = MarkovChainProvider(MCSourceMLE(fsmp, streamTrainSource))
        val wtp = WtProvider(WtSourceMatrix(fsmp, mp, horizon = 10, finalsEnabled = false))
        val pp = ForecasterProvider(ForecasterSourceBuild(fsmp, wtp, horizon = 10, confidenceThreshold = 0.5, maxSpread = 4, method = ForecastMethod.CLASSIFY_NEXTK))

        val erft = ERFTask(
          fsmp             = fsmp,
          pp               = pp,
          predictorEnabled = true,
          finalsEnabled    = false,
          expirationDeadline   = ConfigUtils.defaultExpiration,
          distance         = distance,
          streamSource     = streamTestSource
        )
        val prof = erft.execute()
        prof.printProfileInfo()
      }

      // You should delete the train file here. The ERFTask works in a "chain of command" fashion. Since the
      // SPSTProvider has not been created by passing a SPST object directly to it, the ERFTask will use the
      // SDFAProvider and the stream to train a model upon calling its execute method.
      trainFile.delete
      testFile.delete

      k += 1
  }

  def makeStratifiedSets(
      positives: Iterable[Word[Character]],
      negatives: Iterable[Word[Character]],
      folds: Int
  ): List[(Iterable[Word[Character]], Iterable[Word[Character]], Iterable[Word[Character]])] = {

    val numOfPos = positives.size
    val numOfNeg = negatives.size

    println(s"Found ${numOfPos} positives and ${numOfNeg} negatives.")

    if (numOfPos < folds || numOfNeg < folds) {
      println(s"Positive examples ($numOfPos) or negative examples ($numOfNeg) are less than the given folds!")
      sys.exit(1)
    }

    val posPerFold = numOfPos / folds
    val negPerFold = numOfNeg / folds

    println(s"Each fold will use ${posPerFold} positives and ${negPerFold} negatives.")

    positives.grouped(posPerFold).toList.zip(negatives.grouped(negPerFold).toList).map {
      case (posSet, negSet) =>
        (positives.filter(x => !posSet.exists(_ == x)), negatives.filter(x => negSet.exists(_ == x)), (posSet ++ negSet))
    }
  }

  def fromStringsToCSVStream(simulations: Iterable[Word[Character]], name: String): File = {
    val dataFile = new File(s"$name.csv")
    val output = new PrintStream(dataFile)

    var t = 1
    simulations.foreach { simulation =>
      simulation.asList.asScala.foreach { symbol =>
        output.println(s"$symbol,$t")
        t += 1
      }
      output.println("r,-1")
    }
    output.close()
    dataFile
  }

  def readData(filename: String): Iterable[Word[Character]] = {

    val source = Source.fromFile(filename)
    val words = source.getLines.map { line =>
      Word.fromCharSequence(line)
    }.toList
    source.close

    words
  }
}
