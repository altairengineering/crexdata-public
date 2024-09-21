package workflow.task.engineTask

import com.typesafe.scalalogging.LazyLogging
import engine.{ERFEngine, OOFERFEngine}
import model.waitingTime.ForecastMethod.ForecastMethod
import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.apache.kafka.clients.producer.KafkaProducer
import profiler.WtProfiler
import stream.source.StreamSource
import ui.ConfigUtils
import utils.avro.AvroProducer
import workflow.provider.source.forecaster.ForecasterSourceRandom
import workflow.provider.{FSMProvider, ForecasterProvider}
import workflow.task.Task

object OOFERFTask {
  /**
    * Constructor for ERF task.
    *
    * @param fsmp The provider for the FSM.
    * @param pp The provider for the predictor.
    * @param predictorEnabled If true, forecasting is enabled. If false, only recognition.
    * @param finalsEnabled If true, forecasts from final states are emitted (the distances provided must also include the
    *                      finals).
    * @param expirationDeadline The time a FSM run is allowed to live before being recycled. See engine.ERFEngine.
    * @param distance The distances to determine when forecasts are allowed to be emitted. See engine.ERFEngine.
    * @param streamSource The source for the event stream.
    * @param collectStats If true, recognition and forecasting statistics are enabled (lowers performance).
    * @param show Determines whether complex event matches and forecasts are to be displayed or not.
    * @return An ERT task.
    */
  def apply(
             fsmp: FSMProvider,
             pp: ForecasterProvider,
             predictorEnabled: Boolean,
             finalsEnabled: Boolean,
             expirationDeadline: Long,
             distance: (Double, Double), horizon: Int, maxSpread: Int,
             method: ForecastMethod,
             streamSource: StreamSource,
             collectStats: Boolean,
             show: Boolean,
             outputStreamProducer: AvroProducer,
             statsProducer: AvroProducer,
             statsReportingDistance: Long
           ): OOFERFTask = new OOFERFTask(
    fsmp,
    pp,
    predictorEnabled,
    finalsEnabled,
    expirationDeadline,
    distance, horizon, maxSpread, method,
    streamSource,
    collectStats,
    show,
    outputStreamProducer,
    statsProducer, statsReportingDistance)

  /**
    * Constructor for ERF task.
    * Defaults values for predictorEnabled, expirationDeadline, distance, collectStats.
    *
    * @param fsmp The provider for the FSM.
    * @param pp The provider for the predictor.
    * @param finalsEnabled If true, forecasts from final states are emitted (the distances provided must also include the
    *                      finals).
    * @param streamSource The source for the event stream.
    * @return An ERF task.
    */
  def apply(
             fsmp: FSMProvider,
             pp: ForecasterProvider,
             finalsEnabled: Boolean,
             streamSource: StreamSource,
             horizon: Int, maxSpread: Int,
             method: ForecastMethod,
             outputStreamProducer: AvroProducer,
             statsProducer: AvroProducer,
             statsReportingDistance: Long
           ): OOFERFTask = new OOFERFTask(
    fsmp,
    pp,
    predictorEnabled = ConfigUtils.defaultPredictorEnabled,
    finalsEnabled,
    expirationDeadline = ConfigUtils.defaultExpiration,
    distance       = ConfigUtils.defaultDistance, horizon: Int, maxSpread: Int,
    method: ForecastMethod,
    streamSource,
    collectStats = ConfigUtils.defaultCollectStats,
    show = ConfigUtils.defaultShowMatchesForecasts,
    outputStreamProducer,
    statsProducer, statsReportingDistance)

  /**
    * Constructor for ERF task.
    * Defaults values for predictorEnabled, finalsEnabled, expirationDeadline, distance, collectStats.
    *
    * @param fsmp The provider for the FSM.
    * @param pp The provider for the predictor.
    * @param streamSource The source for the event stream.
    * @param show Determines whether complex event matches and forecasts are to be displayed or not.
    * @return An ERF task.
    */
  def apply(
             fsmp: FSMProvider,
             pp: ForecasterProvider,
             streamSource: StreamSource, horizon: Int, maxSpread: Int,
             method: ForecastMethod,
             show: Boolean,
             outputStreamProducer: AvroProducer,
             statsProducer: AvroProducer,
             statsReportingDistance: Long
           ): OOFERFTask = new OOFERFTask(
    fsmp,
    pp,
    predictorEnabled = ConfigUtils.defaultPredictorEnabled,
    finalsEnabled    = ConfigUtils.defaultFinalsEnabled,
    expirationDeadline   = ConfigUtils.defaultExpiration,
    distance         = ConfigUtils.defaultDistance, horizon: Int, maxSpread: Int,
    method: ForecastMethod,
    streamSource,
    collectStats = ConfigUtils.defaultCollectStats,
    show,
    outputStreamProducer,
    statsProducer,statsReportingDistance)

  /**
    * Constructor for ERF task.
    * Defaults values for predictorEnabled, finalsEnabled, expirationDeadline, collectStats.
    *
    * @param fsmp The provider for the FSM.
    * @param pp The provider for the predictor.
    * @param distance The distances to determine when forecasts are allowed to be emitted. See engine.ERFEngine.
    * @param streamSource The source for the event stream.
    * @return An ERF task.
    */
  def apply(
             fsmp: FSMProvider,
             pp: ForecasterProvider,
             distance: (Double, Double), horizon: Int, maxSpread: Int,
             method: ForecastMethod,
             streamSource: StreamSource,
             outputStreamProducer: AvroProducer,
             statsProducer: AvroProducer,
             statsReportingDistance: Long
           ): OOFERFTask = new OOFERFTask(
    fsmp,
    pp,
    predictorEnabled = ConfigUtils.defaultPredictorEnabled,
    finalsEnabled    = ConfigUtils.defaultFinalsEnabled,
    expirationDeadline   = ConfigUtils.defaultExpiration,
    distance, horizon: Int, maxSpread: Int,
    method: ForecastMethod,
    streamSource,
    collectStats = ConfigUtils.defaultCollectStats,
    show = ConfigUtils.defaultShowMatchesForecasts,
    outputStreamProducer,
    statsProducer,statsReportingDistance)


  /**
    * Constructor for ERF task.
    * Default value for collectStats.
    *
    * @param fsmp The provider for the FSM.
    * @param pp The provider for the predictor.
    * @param predictorEnabled If true, forecasting is enabled. If false, only recognition.
    * @param finalsEnabled If true, forecasts from final states are emitted (the distances provided must also include the
    *                      finals).
    * @param expirationDeadline The time a FSM run is allowed to live before being recycled. See engine.ERFEngine.
    * @param distance The distances to determine when forecasts are allowed to be emitted. See engine.ERFEngine.
    * @param streamSource The source for the event stream.
    * @return An ERT task.
    */
  def apply(
             fsmp: FSMProvider,
             pp: ForecasterProvider,
             predictorEnabled: Boolean,
             finalsEnabled: Boolean,
             expirationDeadline: Long,
             distance: (Double, Double), horizon: Int, maxSpread: Int,
             method: ForecastMethod,
             streamSource: StreamSource,
             outputStreamProducer: AvroProducer,
             statsProducer: AvroProducer, statsReportingDistance: Long
           ): OOFERFTask = new OOFERFTask(
    fsmp,
    pp,
    predictorEnabled,
    finalsEnabled,
    expirationDeadline,
    distance, horizon: Int, maxSpread: Int,
    method: ForecastMethod,
    streamSource,
    collectStats = ConfigUtils.defaultCollectStats,
    show = ConfigUtils.defaultShowMatchesForecasts,
    outputStreamProducer,
    statsProducer, statsReportingDistance)


  /**
    * Constructor for ERF task. To be used for recognition. Forecasting is disabled.
    * Default value for expirationDeadline, collectStats.
    *
    * @param fsmp The provider for the FSM.
    * @param streamSource The source for the event stream.
    * @return
    */
  def apply(
             fsmp: FSMProvider,
             horizon: Int, maxSpread: Int,
             method: ForecastMethod,
             streamSource: StreamSource,
             outputStreamProducer: AvroProducer,
             statsProducer: AvroProducer,
             statsReportingDistance: Long
           ): OOFERFTask = new OOFERFTask(
    fsmp,
    ForecasterProvider(new ForecasterSourceRandom(fsmp, 1)),
    predictorEnabled = false,
    finalsEnabled = false,
    expirationDeadline =ConfigUtils.defaultExpiration,
    distance = ConfigUtils.defaultDistance, horizon: Int, maxSpread: Int,
    method: ForecastMethod,
    streamSource,
    collectStats = ConfigUtils.defaultCollectStats,
    show = ConfigUtils.defaultShowMatchesForecasts,
    outputStreamProducer,
    statsProducer,statsReportingDistance)
  /**
    * Constructor for ERF task. To be used for recognition. Forecasting is disabled.
    * Default value for expirationDeadline, collectStats.
    *
    * @param fsmp The provider for the FSM.
    * @param streamSource The source for the event stream.
    * @param show Determines whether complex event matches and forecasts are to be displayed or not.
    * @return
    */
  def apply(
             fsmp: FSMProvider,
             streamSource: StreamSource,
             horizon: Int, maxSpread: Int,
             method: ForecastMethod,
             show: Boolean,
             outputStreamProducer: AvroProducer,
             statsProducer: AvroProducer,
             statsReportingDistance: Long
           ): OOFERFTask = new OOFERFTask(
    fsmp,
    ForecasterProvider(new ForecasterSourceRandom(fsmp, 1)),
    predictorEnabled = false,
    finalsEnabled = false,
    expirationDeadline =ConfigUtils.defaultExpiration,
    distance = ConfigUtils.defaultDistance, horizon: Int, maxSpread: Int,
    method: ForecastMethod,
    streamSource,
    collectStats = ConfigUtils.defaultCollectStats, show,
    outputStreamProducer,
    statsProducer,statsReportingDistance)
}

/**
  * Task that runs event recognition and forecasting.
  *
  * @param fsmp The provider for the FSM.
  * @param pp The provider for the predictor.
  * @param predictorEnabled If true, forecasting is enabled. If false, only recognition.
  * @param finalsEnabled If true, forecasts from final states are emitted (the distances provided must also include the
  *                      finals).
  * @param expirationDeadline The time a FSM run is allowed to live before being recycled. See engine.ERFEngine.
  * @param distance The distances to determine when forecasts are allowed to be emitted. See engine.ERFEngine.
  * @param streamSource The source for the event stream.
  * @param collectStats If true, recognition and forecasting statistics are enabled (lowers performance).
  * @param show Determines whether complex event matches and forecasts are to be displayed or not.
  */
class OOFERFTask  (
                    fsmp: FSMProvider,
                    pp: ForecasterProvider,
                    predictorEnabled: Boolean,
                    finalsEnabled: Boolean,
                    expirationDeadline: Long,
                    distance: (Double, Double), horizon: Int, maxSpread: Int,
                    method: ForecastMethod,
                    streamSource: StreamSource,
                    collectStats: Boolean,
                    show: Boolean,
                    outputStreamProducer: AvroProducer,
                    statsProducer: AvroProducer,
                    statsReportingDistance: Long
                  ) extends ERFTask(
  fsmp: FSMProvider,
  pp: ForecasterProvider,
  predictorEnabled: Boolean,
  finalsEnabled: Boolean,
  expirationDeadline: Long,
  distance: (Double, Double),
  streamSource: StreamSource,
  collectStats: Boolean,
  show: Boolean) {

  private val engine = {
    OOFERFEngine(fsmp, pp, predictorEnabled,
      expirationDeadline, collectStats, finalsEnabled,
      distance, horizon, maxSpread, method, show,
      outputStreamProducer,
      statsProducer,
      statsReportingDistance
    )
  }

  /**
    * Sends events to the engine.
    *
    * @return A profiler with statistics.
    */
  override def execute(): WtProfiler = {
    println("Executing...")
    logger.info("Executing ERF task...")
    streamSource.emitEventsToListener(engine)
    val profiler = engine.getProfiler
    logger.info("done.")
    profiler
  }

  override def getEngine: ERFEngine = engine

}
