package scripts.learning

import com.github.tototoshi.csv.{CSVReader, CSVWriter}
import com.typesafe.scalalogging.LazyLogging
import stream.StreamFactory
import model.waitingTime.ForecastMethod
import ui.ConfigUtils
import workflow.provider.{FSMProvider, ForecasterProvider, SDFAProvider, SPSTProvider, WtProvider}
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.source.sdfa.{SDFASourceDirectI, SDFASourceFromSRE}
import workflow.provider.source.spst.{SPSTSourceDirectI, SPSTSourceFromSDFA}
import workflow.provider.source.wt.{WtSourceDirect, WtSourceSPST}
import workflow.task.engineTask.ERFTask

object ProofOfConcept3 extends App with LazyLogging {
  final val home = System.getenv("WAYEB_HOME")
  final val dataDirs = List(
    /*System.getenv("HOME") + "/data/bio/normalized/symbolic_sims_12_symbols_norm/",
    System.getenv("HOME") + "/data/bio/normalized/symbolic_sims_8_symbols_norm/",
    System.getenv("HOME") + "/data/bio/normalized/symbolic_sims_12_symbols_norm/",
    System.getenv("HOME") + "/data/bio/normalized/symbolic_sims_16_symbols_norm/"*/

    //home + "/data/bio/symbolic_sims_8_symbols/",
    home + "/scripts/data/bio/symbolic_sims_8_symbols/" //,
    //home + "/data/bio/symbolic_sims_16_symbols/"

  /*home + "/data/bio/symbolic_sims_12_symbols/",
    home + "/data/bio/symbolic_sims_12_symbols/"*/
  )
  final val domain = "bio"
  final val patternFile = home + "/patterns/bio/patternNotInteresting.sre"
  final val declarationsFiles = List(
    /*home + "/patterns/bio/declarationsBTAliveNo.sre",
    home + "/patterns/bio/declarations8AliveSymbols.sre",
    home + "/patterns/bio/declarations12AliveSymbols.sre",
    home + "/patterns/bio/declarations16AliveSymbols.sre"*/

    //home + "/patterns/bio/declarations8AliveSymbols.sre",
    home + "/patterns/bio/declarationsBTAliveNo.sre"//,
    //home + "/patterns/bio/declarations16AliveSymbols.sre"

  /*home + "/patterns/bio/declarationsBTAliveNo.sre",
    home + "/patterns/bio/declarations12AliveSymbols.sre"*/
  )
  final val bins = List(12.0) //List(8.6, 12.0, 16.6) //List(12.0,8.6,12.6,16.6) //List(12.0, 12.6)
  final val predictionThresholds = List(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

  //final val horizon = 15
  //final val maxSpread = 15
  //final val distance = (0.0001, 1.0)
  final val method = ForecastMethod.CLASSIFY_NEXTK
  final val distances = List(20.0) //List(5.0, 10.0, 20.0)
  final val order = 6

  // hyper-parameters
  final val pMin = 0.001
  final val alpha = 0.0
  final val gamma = 0.001
  final val r = 1.05
  final val cutoffThreshold = 0.01

  final val foldsNo = 5

  final val statsPerFoldFile = home + "/results/bio/statsPerFold.csv"
  final val statsAverageFile = home + "/results/bio/statsAverage.csv"
  final val statsRocFile = home + "/results/bio/statsRoc.csv"

  runExperiments()

  processStats()

  def processStats(): Unit = {
    val reader = CSVReader.open(statsPerFoldFile)
    val writerAvg = CSVWriter.open(statsAverageFile)
    writerAvg.writeRow(List(
      "bins",
      "distance",
      "threshold",
      "tp",
      "tn",
      "fp",
      "fn",
      "precision",
      "recall",
      "f1",
      "npv",
      "specificity",
      "accuracy",
      "mcc"
    ))
    val writerRoc = CSVWriter.open(statsRocFile)
    val lines = reader.all().drop(1)
    distances.foreach(distance => {
      bins.foreach(binsNo => {
        var recalls: List[Double] = List.empty
        var fallouts: List[Double] = List.empty
        predictionThresholds.foreach(threshold => {
          var tp = 0
          var tn = 0
          var fp = 0
          var fn = 0
          (1 to foldsNo).foreach(fold => {
            val row = lines.filter(line => line.head.toDouble == binsNo & line(1).toInt == fold & line(2).toDouble == distance & line(3).toDouble == threshold).head
            tp += row(4).toInt
            tn += row(5).toInt
            fp += row(6).toInt
            fn += row(7).toInt
          })
          val scores = estimateMetrics(tp, tn, fp, fn)
          val row = List(
            binsNo.toString,
            distance.toString,
            threshold.toString,
            tp.toString,
            tn.toString,
            fp.toString,
            fn.toString,
            scores("precision").toString,
            scores("recall").toString,
            scores("f1").toString,
            scores("npv").toString,
            scores("specificity").toString,
            scores("accuracy").toString,
            scores("mcc").toString
          )
          writerAvg.writeRow(row)
          recalls = scores("recall") :: recalls
          fallouts = (1 - scores("specificity") :: fallouts)
        })
        writerRoc.writeRow(List(binsNo.toString, distance.toString) ::: recalls ::: fallouts)
      })
    })
    reader.close()
    writerAvg.close()
    writerRoc.close()
  }

  def runExperiments(): Unit = {
    val writer = CSVWriter.open(statsPerFoldFile)
    writer.writeRow(List(
      "bins",
      "fold",
      "distance",
      "threshold",
      "tp",
      "tn",
      "fp",
      "fn",
      "precision",
      "recall",
      "f1",
      "npv",
      "specificity",
      "accuracy",
      "mcc"
    ))

    declarationsFiles.indices.foreach(index => {
      val declarationsFile = declarationsFiles(index)
      logger.info("Constructing SDFA")
      val sdfaSource = SDFASourceFromSRE(patternFile, ConfigUtils.defaultPolicy, declarationsFile, "withoutsat")
      val sdfap0 = SDFAProvider(sdfaSource)
      val sdfap = SDFAProvider(SDFASourceDirectI(List(sdfap0.provide().head)))

      (1 to foldsNo).foreach(fold => {
        logger.info("\t\t\t Fold " + fold + "\n\n")

        val trainFile = dataDirs(index) + "train" + fold + ".csv"
        val testFile = dataDirs(index) + "test" + fold + ".csv"

        logger.info("Learning prediction suffix tree")
        val streamTrainSource = StreamFactory.getDomainStreamSource(trainFile, domain = domain, List.empty)
        val spstp0 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource, pMin = pMin, alpha = alpha, gamma = gamma, r = r))
        val spstp = SPSTProvider(SPSTSourceDirectI(List(spstp0.provide().head)))
        val fsmp = FSMProvider(spstp)

        distances.foreach(distance => {
          logger.info("Estimating waiting-time distributions for distance=" + distance)
          val wtp0 = WtProvider(WtSourceSPST(
            spstp,
            horizon         = distance.toInt + 5,
            cutoffThreshold = cutoffThreshold,
            distance        = (distance, distance)
          ))
          val wtp = WtProvider(WtSourceDirect(List(wtp0.provide().head)))

          val streamTestSource = StreamFactory.getDomainStreamSource(testFile, domain = domain, List.empty)
          predictionThresholds.foreach(predictionThreshold => {
            logger.info("Estimating predictions for threshold = " + predictionThreshold)
            val pp = ForecasterProvider(ForecasterSourceBuild(
              fsmp,
              wtp,
              horizon             = distance.toInt + 5,
              confidenceThreshold = predictionThreshold,
              maxSpread           = distance.toInt + 5,
              method              = method
            ))
            logger.info("Now running forecasting")
            val erft = ERFTask(
              fsmp             = fsmp,
              pp               = pp,
              predictorEnabled = true,
              finalsEnabled    = false,
              expirationDeadline   = ConfigUtils.defaultExpiration,
              distance         = (distance, distance),
              streamSource     = streamTestSource
            )
            val prof = erft.execute()
            prof.printProfileInfo()

            val row = List(
              bins(index).toString,
              fold.toString,
              distance.toString,
              predictionThreshold.toString,
              prof.getStatFor("tp", 0),
              prof.getStatFor("tn", 0),
              prof.getStatFor("fp", 0),
              prof.getStatFor("fn", 0),
              prof.getStatFor("precision", 0),
              prof.getStatFor("recall", 0),
              prof.getStatFor("f1", 0),
              prof.getStatFor("npv", 0),
              prof.getStatFor("specificity", 0),
              prof.getStatFor("accuracy", 0),
              prof.getStatFor("mcc", 0)
            )
            writer.writeRow(row)
          })
        })
      })
    })
    writer.close()
  }

  def estimateMetrics(
                       tp: Int,
                       tn: Int,
                       fp: Int,
                       fn: Int
                     ): Map[String, Double] = {
    val tpfp = tp + fp
    val precision = if (tpfp != 0) tp.toDouble / tpfp else -1
    val tpfn = tp + fn
    val recall = if (tpfn != 0) tp.toDouble / tpfn else -1
    val f1 = if (precision != -1 & recall != -1) (2 * precision * recall) / (precision + recall) else -1
    val tnfp = tn + fp
    val specificity = if (tnfp != 0) tn.toDouble / tnfp else -1
    val total = tp + tn + fp + fn
    val accuracy = if (total != 0) (tp + tn).toDouble / total else -1
    val tnfn = tn + fn
    val npv = if (tnfn != 0) tn.toDouble / tnfn else -1
    val informedness = if (recall != -1 & specificity != -1) recall + specificity - 1 else -1
    val mcc =
      if (tpfp == 0 | tpfn == 0 | tnfp == 0 | tnfn == 0) 0.0
      else {
        val fdr = 1 - precision
        val fnr = 1 - recall
        val fpr = 1 - specificity
        val fomr = 1 - npv
        math.sqrt(precision * recall * specificity * npv) - math.sqrt(fdr * fnr * fpr * fomr)
      }
    Map(
      "precision" -> precision,
      "recall" -> recall,
      "f1" -> f1,
      "specificity" -> specificity,
      "accuracy" -> accuracy,
      "npv" -> npv,
      "informedness" -> informedness,
      "mcc" -> mcc
    )
  }
}
