package scripts.learning

import java.io.{File, PrintStream}

import scripts.learning.ProofOfConceptMain2_Classification.order
import net.automatalib.words.Word
import net.automatalib.words.impl.Alphabets
import stream.StreamFactory
import model.waitingTime.ForecastMethod
import ui.ConfigUtils
import workflow.provider.source.matrix.MCSourceMLE
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.{FSMProvider, ForecasterProvider, MarkovChainProvider, SDFAProvider, SPSTProvider, WtProvider}
import workflow.provider.source.sdfa.{SDFASourceDirectI, SDFASourceFromSRE, SDFASourceLLDFA}
import workflow.provider.source.spst.{SPSTSourceDirectI, SPSTSourceFromSDFA}
import workflow.provider.source.wt.{WtSourceMatrix, WtSourceSPST}
import workflow.task.engineTask.ERFTask

import scala.collection.JavaConverters._
import scala.io.Source

object ProofOfConceptMain2 extends App {

  final val folds = 5
  final val order = 5
  final val alphabet = Alphabets.characters('a', 'h')
  final val pathToPositiveExamples = "scripts/data/bio/interesting_alive.txt"
  final val pathToNegativeExamples = "scripts/data/bio/not_interesting_alive.txt"
  final val home = System.getenv("WAYEB_HOME")
  final val predictionThreshold = 0.2
  final val horizon = 20
  final val domain = "bio"
  final val maxSpread = 20
  final val method = ForecastMethod.CLASSIFY_NEXTK
  final val distance = (0.0001, 1.0)

  final val pMin = 0.001
  final val alpha = 0.0
  final val gamma = 0.001
  final val r = 1.05
  final val cutoffThreshold = 0.01

  val positiveExamples = readData(pathToPositiveExamples)
  val negativeExamples = readData(pathToNegativeExamples)

  var k = 1
  makeStratifiedSets(positiveExamples, negativeExamples, folds).foreach {
    case (trainPosSet, trainNegSet, testPosSet, testNegSet) =>

      println(s"Fold $k:")

      val learner = RPNI(alphabet)
      val result = learner.train(trainPosSet, trainNegSet)

      // -- Resulted DFA
      println(s"States: {${result.states.mkString(", ")}}")
      println(s"Transitions: {${result.transitions.mkString(", ")}}")
      println(s"Initial State: ${result.initialState}")
      println(s"Final States: {${result.finalStates.mkString(", ")}}")

      // Create the learned automaton
      /*val sdfaSource = SDFASourceLLDFA(
        result.states, result.transitions, result.initialState, result.finalStates, order, streaming = false)
      val sdfap = SDFAProvider(sdfaSource)
      val fsmp = FSMProvider(sdfap)*/
      val sdfaSource = SDFASourceFromSRE(home + "/patterns/bio/pattern.sre", ConfigUtils.defaultPolicy, home + "/patterns/bio/declarations.sre")
      val sdfap0 = SDFAProvider(sdfaSource)
      val sdfa = sdfap0.provide().head
      val sdfap = SDFAProvider(SDFASourceDirectI(List(sdfa)))

      val trainFile = fromStringsToCSVStream(trainPosSet, trainNegSet, "train" + k)
      val streamTrainSource = StreamFactory.getDomainStreamSource(trainFile.getAbsolutePath, domain = domain, List.empty)
      //sys.exit(1)
      /*val mp = MatrixProvider(MatrixSourceMLE(fsmp, streamTrainSource))
      val wtp = WtProvider(WtSourceMatrix(fsmp, mp, horizon = horizon, finalsEnabled = false))
      val pp = PredictorProvider(PredictorSourceBuild(fsmp, wtp,
        horizon = horizon,
        predictionThreshold = predictionThreshold,
        maxSpread = maxSpread,
        method = method))*/
      val spstp0 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource, pMin = pMin, alpha = alpha, gamma = gamma, r = r))
      val spst = spstp0.provide().head
      val spstp = SPSTProvider(SPSTSourceDirectI(List(spst)))
      val fsmp = FSMProvider(spstp)
      val wtp = WtProvider(WtSourceSPST(spstp, horizon = horizon, cutoffThreshold = cutoffThreshold, distance = distance))
      val pp = ForecasterProvider(ForecasterSourceBuild(fsmp, wtp,
                                                      horizon             = horizon,
                                                      confidenceThreshold = predictionThreshold,
                                                      maxSpread           = maxSpread,
                                                      method              = method))

      val testFile = fromStringsToCSVStream(testPosSet, testNegSet, "test" + k)
      val streamTestSource = StreamFactory.getDomainStreamSource(testFile.getAbsolutePath, domain = domain, List.empty)

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

      trainFile.delete
      testFile.delete

      k += 1
  }

  def makeStratifiedSets(
      positives: Iterable[Word[Character]],
      negatives: Iterable[Word[Character]],
      folds: Int
  ): List[(Iterable[Word[Character]], Iterable[Word[Character]], Iterable[Word[Character]], Iterable[Word[Character]])] = {

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
        (positives.filter(x => !posSet.exists(_ == x)),
          negatives.filter(x => negSet.exists(_ == x)),
          posSet,
          negSet
        )
    }
  }

  def fromStringsToCSVStream(
      posSimulations: Iterable[Word[Character]],
      negSimulations: Iterable[Word[Character]],
      name: String
  ): File = {

    val dataFile = new File(s"$name.csv")
    val output = new PrintStream(dataFile)

    var t = 1
    posSimulations.foreach { simulation =>
      simulation.asList.asScala.zip(Vector.fill(simulation.length - 1)(0) :+ 1)
        .foreach { case (symbol, isLast) =>
          output.println(s"$symbol,$t,$isLast")
          t += 1
        }
      output.println("r,-1")
    }
    negSimulations.foreach { simulation =>
      simulation.asList.asScala.foreach { symbol =>
        output.println(s"$symbol,$t,0")
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
