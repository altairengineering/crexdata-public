package ui.demo

import model.waitingTime.ForecastMethod
import stream.StreamFactory
import ui.ConfigUtils
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.source.matrix.MCSourceMLE
import workflow.provider.source.sdfa.SDFASourceFromSRE
import workflow.provider.source.wt.{WtSourceDirect, WtSourceMatrix, WtSourceSPST}
import workflow.provider._
import workflow.provider.source.spst.{SPSTSourceDirectI, SPSTSourceFromSDFA}
import workflow.task.engineTask.ERFTask

object RunSrc extends App {
  final val confidenceThreshold = 0.5
  final val horizon = 50
  final val domain = "maritime"
  final val maxSpread = 5
  final val method = ForecastMethod.CLASSIFY_NEXTK
  final val distance = (0.0001, 1.0)

  final val home = System.getenv("WAYEB_HOME")
  final val dataDir: String = home + "/data/maritime/"
  final val resultsDir: String = home + "/results"
//  final val testDatasetFilename: String = dataDir + "227592820.csv"
//  final val trainDatasetFilename: String = dataDir + "227592820.csv"
    final val testDatasetFilename: String = home + "/1443650401-1459461585_gap1800_interval60_speed1.0_matches101/port/all/folds/fold1_test.csv"
    final val trainDatasetFilename: String = home + "/1443650401-1459461585_gap1800_interval60_speed1.0_matches101/port/all/folds/fold1_test.csv"
  final val patternFile: String = home + "/patterns/maritime/port/pattern.sre"
  final val declarationsFile: String = home + "/patterns/maritime/port/declarationsDistance1.sre"

  // First create the training and test stream sources.
  // For convenience, here we use the same file, but should be different in real experiments.
  val streamTrainSource = StreamFactory.getDomainStreamSource(trainDatasetFilename, domain = domain, List.empty)
  val streamTestSource = StreamFactory.getDomainStreamSource(testDatasetFilename, domain = domain, List.empty)

  // Create a provider for the SDFA
  val sdfap = SDFAProvider(SDFASourceFromSRE(patternFile, ConfigUtils.defaultPolicy, declarationsFile))
  // Wrap a FSM provider around it
  val fsmp = FSMProvider(sdfap)
  // Create a provider for the Markov chain model
  val mp = MarkovChainProvider(MCSourceMLE(fsmp, streamTrainSource))
  // Create a provider for the waiting-time distributions
  val wtp = WtProvider(WtSourceMatrix(fsmp, mp, horizon = horizon, finalsEnabled = false))
  // Create a provider for the forecast intervals
  val pp = ForecasterProvider(ForecasterSourceBuild(
    fsmp,
    wtp,
    horizon             = horizon,
    confidenceThreshold = confidenceThreshold,
    maxSpread           = maxSpread,
    method              = ForecastMethod.CLASSIFY_NEXTK
  ))

  // Now execute recognition and forecasting
  val erft = ERFTask(
    fsmp             = fsmp,
    pp               = pp,
    predictorEnabled = true,
    finalsEnabled    = false,
    expirationDeadline   = ConfigUtils.defaultExpiration,
    distance         = distance,
    streamSource     = streamTestSource,
    collectStats = true,
    show = true
  )
  val prof = erft.execute()
  prof.printProfileInfo()

  val f1score = prof.getStatFor("f1", 0)
  println("\n\n\n\n\n\tF1-score: " + f1score)

//  maxOrder The maximum order of the PST (same for all PSTs).
//  pMin This is the symbol threshold. Symbols with lower probability are discarded (same for all PSTs).
//  alpha Used to calculate the conditional threshold = (1 + alpha) * gammaMin. The conditional on the expanded
//              context must be greater than this threshold (same for all PSTs).
//  gammaMin Used to calculate the conditional threshold = (1 + alpha) * gammaMin. The conditional on the
//                    expanded context must be greater than this threshold (same for all PSTs).
//  r This is the likelihood ratio threshold. Contexts are expanded if the probability ratio of the conditional
//            on the expanded context by the conditional on the original context is greater than this threshold (same
//            for all PSTs).
  //[0.0,1.0]?
  final val pMin = 0.001
  //tests for order 1:
  // alpha  |gamma  |r    | f1
  //  0.0   | 0.001 |1.05 | -1
  //  2.0   | 0.001 |1.05 | -1
  //  0.5   | 0.001 |1.05 | -1
  //  10.0  | 0.001 |1.05 | -1
  //  0.01  | 0.001 |1.05 | -1

  //tests for order 2:
  // alpha  |gamma  |r    | f1
  //  0.0   | 0.001 |1.05 | 0.4355088153417878
  //  1.0   | 0.001 |1.05 | 0.4355088153417878
  //  20    | 0.001 |1.05 | 0.4355088153417878
  //  0.5   | 0.001 |1.05 | 0.4355088153417878
  //  0.3   | 0.001 |1.05 | 0.4355088153417878
  //  0.3   | 0.001 |1.05 | 0.4355088153417878
  //  0.3   | 0.001 |2.05 | -1
  //  0.3   | 0.001 |1.5  | -1
  //  0.3   | 0.001 |1.1  | 0.4355088153417878
  //  0.3   | 0.001 |1.2  | 0.4355088153417878
  //  0.3   | 0.001 |1.4  | 0.4344592500774713
  //  0.3   | 0.001 |1.45  | -1
  //  0.3   | 0.001 |1.42  | 0.4344592500774713
  //  0.3   | 0.001 |1.44  | 0.4344592500774713



  final val alpha = 0.0
  final val gamma = 0.001
  final val r = 1.05
  val spstp1 = SPSTProvider(SPSTSourceFromSDFA(sdfap, 2, streamTrainSource, pMin = pMin, alpha = alpha, gamma = gamma, r = r))
  val spstp = SPSTProvider(SPSTSourceDirectI(List(spstp1.provide().head)))
  val fsmp1 = FSMProvider(spstp)
  val wtp0 = WtProvider(WtSourceSPST(
    spstp,
    horizon         = horizon,
    cutoffThreshold = ConfigUtils.wtCutoffThreshold,
    distance        = distance
  ))
  val wtp1 = WtProvider(WtSourceDirect(List(wtp0.provide().head)))
  val pp1 = ForecasterProvider(ForecasterSourceBuild(
    fsmp1,
    wtp1,
    horizon             = horizon,
    confidenceThreshold = confidenceThreshold,
    maxSpread           = maxSpread,
    method              = method
  ))

  val erft1 = ERFTask(
    fsmp             = fsmp1,
    pp               = pp1,
    predictorEnabled = true,
    finalsEnabled    = false,
    expirationDeadline   = ConfigUtils.defaultExpiration,
    distance         = distance,
    streamSource     = streamTestSource
  )
  val prof1 = erft1.execute()
  prof1.printProfileInfo()

  val f1score1 = prof1.getStatFor("f1", 0)
  println("\n\n\n\n\n\tF1-score: " + f1score1)

}
