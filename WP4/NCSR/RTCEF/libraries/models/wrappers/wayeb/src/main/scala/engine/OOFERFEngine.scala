package engine

import profiler.WtProfiler
import stream.GenericEvent
import com.typesafe.scalalogging.LazyLogging
import db.DBConnector
import fsm.{FSMInterface, SPSTInterface}
import fsm.runtime.{MatchDump, Run, RunPool}
import model.forecaster.runtime.{ForecasterRun, ForecasterRunFactory}
import model.waitingTime.ForecastMethod
import model.waitingTime.ForecastMethod.ForecastMethod
import org.apache.avro.Schema
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import stream.array.EventStream
import stream.source.{EndOfStreamEvent, StreamListener, UpdateModelEvent}
import ui.ConfigUtils
import utils.Progressor
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.source.spst.{SPSTSourceDirectI, SPSTSourceSerialized}
import workflow.provider.source.wt.{WtSourceSPST,WtSourceSerialized}
import workflow.provider.{FSMProvider, ForecasterProvider, SPSTProvider, WtProvider}
import play.api.libs.json.{JsObject, Json}
import utils.avro.AvroProducer
import java.lang.Thread
import java.nio.file.{Paths, Files}

object OOFERFEngine {
  /**
   * Create an engine with some default parameter values (retrieved from application.conf).
   * Only FSM provider and Predictor provider given.
   *
   * @param fsmp         The provider for the FSM(s) that run recognition.
   * @param predProvider The provider for the predictor(s) that emit forecasts.
   * @return The engine with the default parameter values.
   */
  def apply(
             fsmp: FSMProvider,
             predProvider: ForecasterProvider,
             horizon: Int, maxSpread: Int,
             method: ForecastMethod,
             outputStreamProducer: AvroProducer,
             statsProducer: AvroProducer,
             statsReportingDistance: Long
           ): OOFERFEngine = {
    apply(fsmp = fsmp,
      predProvider = predProvider,
      predictorEnabled = ConfigUtils.defaultPredictorEnabled,
      expirationDeadline = ConfigUtils.defaultExpiration,
      collectStats = ConfigUtils.defaultCollectStats,
      finalsEnabled = ConfigUtils.defaultFinalsEnabled,
      distance = ConfigUtils.defaultDistance, horizon: Int, maxSpread: Int,
      method: ForecastMethod,
      show = ConfigUtils.defaultShowMatchesForecasts,
      outputStreamProducer, statsProducer, statsReportingDistance
    )
  }

  /**
   * Constructor for engine.
   *
   * @param fsmp               The provider for the FSM(s) that run recognition.
   * @param predProvider       The provider for the predictor(s) that emit forecasts.
   * @param predictorEnabled   Determine whether the predictor will be enabled or not.
   * @param expirationDeadline Determines when the collection of the runs is triggered.
   * @param collectStats       Boolean parameter that determines whether statistics should be collected and estimated
   *                           in runtime.
   * @param finalsEnabled      Boolean parameter that determines whether the final states of the FSMs should emit forecasts
   *                           as well.
   * @param distance           A parameter (Tuple2, (minDistance,maxDistance)) that determines when forecasts should be emitted.
   * @param show               Determines whether complex event matches and forecasts are to be displayed or not.
   * @return An ERF engine.
   */
  def apply(
             fsmp: FSMProvider,
             predProvider: ForecasterProvider,
             predictorEnabled: Boolean,
             expirationDeadline: Long,
             collectStats: Boolean,
             finalsEnabled: Boolean,
             distance: (Double, Double), horizon: Int, maxSpread: Int,
             method: ForecastMethod,
             show: Boolean,
             outputStreamProducer: AvroProducer,
             statsProducer: AvroProducer,
             statsReportingDistance: Long
           ): OOFERFEngine = {
    new OOFERFEngine(fsmp, predProvider, predictorEnabled, expirationDeadline, collectStats,
      finalsEnabled, distance, horizon, maxSpread, method, show,
      outputStreamProducer, statsProducer, statsReportingDistance)
  }

}

/**
 * This is the main engine that runs recognition and forecasting on a stream.
 * The engine is a stream listener (stream.source.StreamListener) that receives events from
 * a stream source (stream.source.StreamSource).
 *
 * @param fsmProvider        The provider for the FSM(s) that run recognition.
 * @param predProvider       The provider for the predictor(s) that emit forecasts.
 * @param predictorEnabled   Determine whether the predictor will be enabled or not.
 *                           If false, no forecasts will be emitted (only recognition).
 * @param expirationDeadline Determines when the collection of the runs is triggered.
 *                           Collection takes place every expirationTime time units.
 *                           Collections reclaim from memory stale runs that have not been been updated since
 *                           currentTimestamp - expirationTime.
 *                           If -1, no collections take place, i.e., all runs remain in memory "for ever".
 *                           CAUTION: enabling collection of runs may make recognition incomplete, i.e.,
 *                           not all matches will be detected, essentially restricting matches to occur within
 *                           a window of expirationTime time units.
 * @param collectStats       Boolean parameter that determines whether statistics should be collected and estimated
 *                           in runtime. Used mostly for testing and evaluation. In real environments, collecting
 *                           statistics may decrease throughput.
 * @param finalsEnabled      Boolean parameter that determines whether the final states of the FSMs should emit forecasts
 *                           as well. If false, only the non-final states emit forecasts.
 * @param distance           A parameter (Tuple2, (minDistance,maxDistance)) that determines when forecasts should be emitted.
 *
 *                           If minDistance==maxDistance==-1, forecasts are emitted eagerly, whenever possible (i.e., whenever
 *                           the current state of a FSM can provide a valid forecast, forecast interval satisfying the confidence
 *                           threshold and max spread constraints).
 *
 *                           If minDistance!=-1 and minDistance < 1.0 (same for maxDistance), then forecasts are emitted only
 *                           from states whose remaining percentage is between minDistance and maxDistance.
 *                           The remaining percentage of a state is the ratio of its shortest path length to a final divided by
 *                           the maximum shortest path from all states. It gives an estimate of how close a state is to
 *                           completion. A remaining percentage of 0.0 means that the automaton is in a final state. A
 *                           remaining percentage of 1.0 means that it is in one of its start states.
 *
 *                           If minDistance!=-1 and minDistance >= 1.0 (same for maxDistance), forecasts are emitted only when
 *                           the current event's timestamp has a temporal distance from the next CE in the stream that is equal
 *                           to the value of the parameter.
 *                           CAUTION: This setting (with distances>=1.0) should be used only when one pattern is provided via
 *                           fsmProvider. Does not work for multiple patterns. Moreover, the input stream must have been
 *                           pre-processed so that each input event has an extra attribute, called "nextCETimestamp", providing
 *                           the timestamp of the next CE. See fsm.runtime.Run#isEmitting(stream.GenericEvent).
 * @param show               Determines whether complex event matches and forecasts are to be displayed or not.
 */

class OOFERFEngine(
                    givenFSMProvider: FSMProvider,
                    predProvider: ForecasterProvider,
                    predictorEnabled: Boolean,
                    expirationDeadline: Long,
                    collectStats: Boolean,
                    finalsEnabled: Boolean,
                    distance: (Double, Double),
                    horizon: Int, maxSpread: Int,
                    method: ForecastMethod,
                    show: Boolean,
                    outputStreamProducer: AvroProducer,
                    statsProducer: AvroProducer,
                    statsReportingDistance: Long
                  )
  extends ERFEngine(
    givenFSMProvider: FSMProvider,
    predProvider: ForecasterProvider,
    predictorEnabled: Boolean,
    expirationDeadline: Long,
    collectStats: Boolean,
    finalsEnabled: Boolean,
    distance: (Double, Double),
    show: Boolean) {

  private var timeToReport:Long = -1
  private var modelVersion = -1
  private var modelCreationMethod = "initial"
  var runtimeTP = 0
  var runtimeTN = 0
  var runtimeFP = 0
  var runtimeFN = 0
  var prevTimestamp = 0L

  private def reInitializeEngine(predProvider: ForecasterProvider, newFsmProvider: FSMProvider): Unit = {
    logger.info("Reinitializing engine...")

    runPool.shutdown()
    predList = predProvider.provide()
    fsmProvider = newFsmProvider
    fsmList = fsmProvider.provide().zip(predList).map(x => {
      if (distance._1 != -1.0 & distance._1 < 1.0) {
        if (x._1.remainingPercentage.isEmpty) x._1.estimateRemainingPercentage
        x._1
      } else x._1
    })
    runPool = RunPool(fsmList, expirationDeadline, distance, show)
    md = new MatchDump()
    predFactory = ForecasterRunFactory(predList, collectStats, finalsEnabled)
    predictorRuns = fsmList.map(fsm => fsm.getId).
      map(id => id -> List.empty[ForecasterRun]).toMap

    matchesNo = 0
    execTime = 0
    lastCollectTime = 0
    streamSize = 0
    profiler = WtProfiler()

    runtimeTP = 0
    runtimeTN = 0
    runtimeFP = 0
    runtimeFN = 0
    logger.info("Reinitialization complete.")
  }

  /**
   * Processes a single event from the input stream.
   *
   * @param event The current event (last event emitted from the stream source).
   */
  override def newEventEmitted(event: GenericEvent): Unit = {
    event match {
      /**
       * Every stream must contain an EndOfStreamEvent as its last event.
       * Upon the arrival of such an event, statistics are estimated and provided through a profiler.
       */
      case _: EndOfStreamEvent => {
        profiler = shutdown()
        logger.info("Use getProfiler to retrieve stats.")
      }
      case _: UpdateModelEvent => {
        val modelPath = event.getValueOf("Path").asInstanceOf[String]

        val confidenceThreshold = event.getValueOf("confidenceThreshold").asInstanceOf[String].toDouble
        modelVersion = event.getValueOf("id").asInstanceOf[String].toInt
        modelCreationMethod = event.getValueOf("creationMethod").asInstanceOf[String]
        val fsmp = FSMProvider(SPSTProvider(SPSTSourceSerialized(modelPath)))

        val wtdFileExists = Files.exists(Paths.get(modelPath+".wtd"))

        val wtp = wtdFileExists match {
          case true => WtProvider(WtSourceSerialized(modelPath+".wtd"))
          case false => {
            val spsti = fsmp.provide().map(x => x.asInstanceOf[SPSTInterface])
            val spstp = SPSTProvider(SPSTSourceDirectI(spsti))

            WtProvider(
              WtSourceSPST(
                spstp,
                horizon = horizon,
                cutoffThreshold = ConfigUtils.wtCutoffThreshold,
                distance = distance
              )
            )
          }
        }

        val pp = ForecasterProvider(
          ForecasterSourceBuild(
            fsmp,
            wtp,
            horizon = horizon,
            confidenceThreshold = confidenceThreshold,
            maxSpread = maxSpread,
            method = method
          )
        )

        reInitializeEngine(pp, fsmp)

        logger.info("Updating engine")
      }
      case _ => {
        val currentTime = event.timestamp

        if (timeToReport == -1){
          timeToReport = currentTime + statsReportingDistance
        }else if (timeToReport > 0){
          if (currentTime >= timeToReport){
            timeToReport = currentTime + statsReportingDistance
            val (locked, unlocked) = runPool.getSize
            val collectors = predictorRuns.map(p => (p._1, p._2.map(pr => pr.getCollector)))
            profiler.setGlobal(
              streamSize   = streamSize,
              execTime     = execTime,
              matchesNo    = matchesNo,
              lockedRuns   = locked,
              unlockedRuns = unlocked,
              matchDump    = md
            )
//            matchesNo = 0
//            execTime = 0
//            lastCollectTime = 0
//            streamSize = 0
            if (predictorEnabled) profiler.createEstimators(collectors)
            profiler.estimateStats()
            profiler.printPatternStats()

            val matrix = profiler.getTPTNFPFN(0) // TODO Hardcoded :'(

            val tp = matrix(0).toInt
            val tn = matrix(1).toInt
            val fp = matrix(2).toInt
            val fn = matrix(3).toInt

            val batchTP = tp - runtimeTP
            val batchTN = tn - runtimeTN
            val batchFP = fp - runtimeFP
            val batchFN = fn - runtimeFN

            runtimeTP = tp
            runtimeTN = tn
            runtimeFP = fp
            runtimeFN = fn

            val runtimeScores = utils.Scores.getMetrics(runtimeTP, runtimeTN, runtimeFP, runtimeFN)
            val batchScores = utils.Scores.getMetrics(batchTP, batchTN, batchFP, batchFN)
            val runtimeMetrics = Json.obj(
              "tp" -> tp,
              "tn" -> tn,
              "fp" -> fp,
              "fn" -> fn,
              "mcc" -> runtimeScores.get("mcc"),
              "precision" -> runtimeScores.get("precision"),
              "recall" -> runtimeScores.get("recall"),
              "f1"-> runtimeScores.get("f1")
            ).toString()
            println(runtimeScores)
            println(batchScores)
            val batchMetrics = Json.obj(
              "tp" -> batchTP,
              "tn" -> batchTN,
              "fp" -> batchFP,
              "fn" -> batchFN,
              "mcc" -> batchScores.get("mcc"),
              "precision" -> batchScores.get("precision"),
              "recall" -> batchScores.get("recall"),
              "f1"-> batchScores.get("f1")
            ).toString()
            val statsKey = new GenericData.Record(statsProducer.schemas("key"))
            val statsValue = new GenericData.Record(statsProducer.schemas("value"))
            statsKey.put("timestamp", prevTimestamp)
            statsValue.put("runtime_metrics", runtimeMetrics)
            statsValue.put("batch_metrics", batchMetrics)
            statsValue.put("model_version", modelVersion)
            statsValue.put("timestamp", prevTimestamp)
            statsValue.put("creation_method", modelCreationMethod)
            statsProducer.send(statsKey, statsValue)
            Thread.sleep(10000) // TODO FIX LATER :)
          }
        }
        streamSize += 1
        processEvent(event)
        prevTimestamp = currentTime
      }
    }
  }
}