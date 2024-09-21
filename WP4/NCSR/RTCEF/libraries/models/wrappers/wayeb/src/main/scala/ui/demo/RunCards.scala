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
import org.scalameter._
import scala.math._
import scala.collection.JavaConverters._

class RunCards {
  final val horizon = 50
  final val domain = "cards"
  final val maxSpread = 5
  final val method = ForecastMethod.CLASSIFY_NEXTK
  final val distance = (0.0001, 1.0)


  final val home = "/media/mpits/Ananke/Wayeb-OFO/Wayeb/Wayeb"
//  final val dataDir: String = home + "/data/maritime/"
//  final val resultsDir: String = home + "/results"
  final val pattDclDir: String = home + "/patterns/cards/increasing/"
  final val dataPath:String = "/data/cards/enriched/increasing/"
  val alpha = 0.0
  val r = 1.05

  def get_score_mult(seed: Int, order: Int, runConfidenceThreshold: Double, pMin: Double, gamma: Double, objective_func: String, cross_set: Int): java.util.List[Double] = {

    val (dataset_dir,patternFile,declarationsFile) = seed match{
      case 1234 => {
        (dataPath, pattDclDir + "/pattern.sre", pattDclDir + "/declarations.sre")
      }
      case 1280 => {
        (dataPath, pattDclDir + "/pattern.sre", pattDclDir + "/declarations.sre")
      }
      case 1300 => {
        (dataPath, pattDclDir + "/pattern.sre", pattDclDir + "/declarations.sre")
      }
      case 1400 => {
        (dataPath, pattDclDir + "/pattern.sre", pattDclDir + "/declarations.sre")
      }
      case 1500 => {
        (dataPath, pattDclDir + "/pattern.sre", pattDclDir + "/declarations.sre")
      }
      case _ => throw new IllegalArgumentException("Seed out of limits: " + seed)
    }

    val (testDatasetFilename, trainDatasetFilename) = cross_set match {
      case 0 => {
        (home + dataset_dir + "/folds/fold1_test.csv", home + dataset_dir + "/folds/fold1_train.csv")
      }
      case 1 => {
        (home + dataset_dir + "/folds/fold2_test.csv", home + dataset_dir + "/folds/fold2_train.csv")
      }
      case 2 => {
        (home + dataset_dir + "/folds/fold3_test.csv", home + dataset_dir + "/folds/fold3_train.csv")
      }
      case 3 => {
        (home + dataset_dir + "/folds/fold4_test.csv", home + dataset_dir + "/folds/fold4_train.csv")
      }

      case _ => throw new IllegalArgumentException("Cross number out of limits: " + cross_set)
    }

    val streamTrainSource = StreamFactory.getDomainStreamSource(trainDatasetFilename, domain = domain, List.empty)
    val streamTestSource = StreamFactory.getDomainStreamSource(testDatasetFilename, domain = domain, List.empty)
    objective_func match {
      case "comb" => {
        val training_time = config(
          //      Key.exec.independentSamples->2,
          //      Key.exec.minWarmupRuns -> 20,
          //          Key.exec.maxWarmupRuns -> 10,
          Key.exec.benchRuns -> 5,
          Key.verbose -> false,
          //      Key.exec.reinstantiation.frequency -> 2,
          Key.exec.outliers.suspectPercent -> 20,
          Key.exec.outliers.covMultiplier -> 1.1
        ) withWarmer {
          new Warmer.Default
        } withMeasurer {
          new Measurer.Default
        } measure {

          val sdfap = SDFAProvider(SDFASourceFromSRE(patternFile, ConfigUtils.defaultPolicy, declarationsFile))

          //          val alpha = 0.0
          //          val r = 1.05
          val spstp1 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource, pMin = pMin, alpha = alpha, gamma = gamma, r = r))
          val spstp = SPSTProvider(SPSTSourceDirectI(List(spstp1.provide().head)))
          val fsmp1 = FSMProvider(spstp)
          val wtp0 = WtProvider(WtSourceSPST(
            spstp,
            horizon = horizon,
            cutoffThreshold = ConfigUtils.wtCutoffThreshold,
            distance = distance
          ))
          val wtp1 = WtProvider(WtSourceDirect(List(wtp0.provide().head)))
          val pp1 = ForecasterProvider(ForecasterSourceBuild(
            fsmp1,
            wtp1,
            horizon = horizon,
            confidenceThreshold = runConfidenceThreshold,
            maxSpread = maxSpread,
            method = method
          ))
        }
        val tt = training_time.value

        val sdfap = SDFAProvider(SDFASourceFromSRE(patternFile, ConfigUtils.defaultPolicy, declarationsFile))
        val spstp1 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource, pMin = pMin, alpha = alpha, gamma = gamma, r = r))
        val spstp = SPSTProvider(SPSTSourceDirectI(List(spstp1.provide().head)))
        val fsmp1 = FSMProvider(spstp)
        val wtp0 = WtProvider(WtSourceSPST(
          spstp,
          horizon = horizon,
          cutoffThreshold = ConfigUtils.wtCutoffThreshold,
          distance = distance
        ))
        val wtp1 = WtProvider(WtSourceDirect(List(wtp0.provide().head)))
        val pp1 = ForecasterProvider(ForecasterSourceBuild(
          fsmp1,
          wtp1,
          horizon = horizon,
          confidenceThreshold = runConfidenceThreshold,
          maxSpread = maxSpread,
          method = method
        ))
        val erft1 = ERFTask(
          fsmp = fsmp1,
          pp = pp1,
          predictorEnabled = true,
          finalsEnabled = false,
          expirationDeadline = ConfigUtils.defaultExpiration,
          distance = distance,
          streamSource = streamTestSource
        )

        val prof1 = erft1.execute()

        val tp = prof1.getStatFor("tp", 0).toDouble
        val tn = prof1.getStatFor("tn", 0).toDouble
        val fp = prof1.getStatFor("fp", 0).toDouble
        val fn = prof1.getStatFor("fn", 0).toDouble

        val resultsList = List(tp,tn,fp,fn,tt)
        resultsList.asJava
        //      List(mcc,tt)
      }
      case _ => throw new IllegalArgumentException("Stat not recognized: " + objective_func)
    }
  }

  def get_score_mult_exh(order: Int, runConfidenceThreshold: Double, pMin: Double, gamma: Double, objective_func: String, cross_set: Int): java.util.List[Double] = {

    val (dataset_dir,patternFile,declarationsFile) = order match{
      case 1 => {
        (dataPath, pattDclDir + "/pattern.sre", pattDclDir + "/declarations.sre")
      }
      case 2 => {
        (dataPath, pattDclDir + "/pattern.sre", pattDclDir + "/declarations.sre")
      }
      case 3 => {
        (dataPath, pattDclDir + "/pattern.sre", pattDclDir + "/declarations.sre")
      }
      case 4 => {
        (dataPath, pattDclDir + "/pattern.sre", pattDclDir + "/declarations.sre")
      }

      case _ => throw new IllegalArgumentException("Order out of range: " + order)
    }

    val (testDatasetFilename, trainDatasetFilename) = cross_set match {

      case 0 => {
        (home + dataset_dir + "/folds/fold1_test.csv", home + dataset_dir + "/folds/fold1_train.csv")
      }
      case 1 => {
        (home + dataset_dir + "/folds/fold2_test.csv", home + dataset_dir + "/folds/fold2_train.csv")
      }
      case 2 => {
        (home + dataset_dir + "/folds/fold3_test.csv", home + dataset_dir + "/folds/fold3_train.csv")
      }
      case 3 => {
        (home + dataset_dir + "/folds/fold4_test.csv", home + dataset_dir + "/folds/fold4_train.csv")
      }
      case _ => throw new IllegalArgumentException("Cross number out of limits: " + cross_set)
    }

    val streamTrainSource = StreamFactory.getDomainStreamSource(trainDatasetFilename, domain = domain, List.empty)
    val streamTestSource = StreamFactory.getDomainStreamSource(testDatasetFilename, domain = domain, List.empty)
    objective_func match {
      case "comb" => {
        val training_time = config(
          //      Key.exec.independentSamples->2,
          //      Key.exec.minWarmupRuns -> 20,
          //          Key.exec.maxWarmupRuns -> 10,
          Key.exec.benchRuns -> 5,
          Key.verbose -> true,
          //      Key.exec.reinstantiation.frequency -> 2,
          Key.exec.outliers.suspectPercent -> 20,
          Key.exec.outliers.covMultiplier -> 1.1
        ) withWarmer {
          new Warmer.Default
        } withMeasurer {
          new Measurer.Default
        } measure {

          val sdfap = SDFAProvider(SDFASourceFromSRE(patternFile, ConfigUtils.defaultPolicy, declarationsFile))

          //          val alpha = 0.0
          //          val r = 1.05
          val spstp1 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource, pMin = pMin, alpha = alpha, gamma = gamma, r = r))
          val spstp = SPSTProvider(SPSTSourceDirectI(List(spstp1.provide().head)))
          val fsmp1 = FSMProvider(spstp)
          val wtp0 = WtProvider(WtSourceSPST(
            spstp,
            horizon = horizon,
            cutoffThreshold = ConfigUtils.wtCutoffThreshold,
            distance = distance
          ))
          val wtp1 = WtProvider(WtSourceDirect(List(wtp0.provide().head)))
          val pp1 = ForecasterProvider(ForecasterSourceBuild(
            fsmp1,
            wtp1,
            horizon = horizon,
            confidenceThreshold = runConfidenceThreshold,
            maxSpread = maxSpread,
            method = method
          ))
        }
        val tt = training_time.value

        val sdfap = SDFAProvider(SDFASourceFromSRE(patternFile, ConfigUtils.defaultPolicy, declarationsFile))
        val spstp1 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource, pMin = pMin, alpha = alpha, gamma = gamma, r = r))
        val spstp = SPSTProvider(SPSTSourceDirectI(List(spstp1.provide().head)))
        val fsmp1 = FSMProvider(spstp)
        val wtp0 = WtProvider(WtSourceSPST(
          spstp,
          horizon = horizon,
          cutoffThreshold = ConfigUtils.wtCutoffThreshold,
          distance = distance
        ))
        val wtp1 = WtProvider(WtSourceDirect(List(wtp0.provide().head)))
        val pp1 = ForecasterProvider(ForecasterSourceBuild(
          fsmp1,
          wtp1,
          horizon = horizon,
          confidenceThreshold = runConfidenceThreshold,
          maxSpread = maxSpread,
          method = method
        ))
        val erft1 = ERFTask(
          fsmp = fsmp1,
          pp = pp1,
          predictorEnabled = true,
          finalsEnabled = false,
          expirationDeadline = ConfigUtils.defaultExpiration,
          distance = distance,
          streamSource = streamTestSource
        )

        val prof1 = erft1.execute()

        val tp = prof1.getStatFor("tp", 0).toDouble
        val tn = prof1.getStatFor("tn", 0).toDouble
        val fp = prof1.getStatFor("fp", 0).toDouble
        val fn = prof1.getStatFor("fn", 0).toDouble

        val resultsList = List(tp,tn,fp,fn,tt)
        resultsList.asJava
        //      List(mcc,tt)
      }
      case _ => throw new IllegalArgumentException("Stat not recognized: " + objective_func)
    }
  }
}

