package utils

import fsm.SPSTInterface
import model.waitingTime.{ForecastMethod, WtDistribution}
import org.apache.avro.generic.GenericRecord
import org.apache.kafka.clients.consumer.KafkaConsumer
import org.scalameter._
import stream.StreamFactory
import ui.ConfigUtils
import workflow.provider._
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.source.sdfa.SDFASourceFromSRE
import workflow.provider.source.spst.{SPSTSourceDirectI, SPSTSourceFromSDFA, SPSTSourceSerialized}
import workflow.provider.source.wt.{WtSourceDirect, WtSourceSPST, WtSourceSerialized}
import workflow.task.engineTask.{ERFTask, OOFERFTask}
import avro.{AvroProducer, AvroConsumer}
import org.apache.avro.Schema
import org.apache.kafka.clients.producer.KafkaProducer
import stream.domain.maritime.MaritimeWAStreamSourceOOF
import stream.domain.cards.TrxStreamSourceOOF
import stream.source.StreamSource

import java.util.{Collections, Properties}
import java.nio.file.{Paths, Files}
import scala.collection.JavaConverters._

class WrappedMethods4Python {
  final val horizon = 50
  final val maxSpread = 5
  final val method = ForecastMethod.CLASSIFY_NEXTK

  val alpha = 0.0
  val r = 1.05

  def trainAndSave(TrainPath: String,
                   PatternPath: String,
                   DeclarationsPath: String,
                   SavePath: String,
                   domain: String,
                   distanceMin: Double,
                   distanceMax: Double,
                   order: Int,
                   pMin: Double,
                   gamma: Double,
                   objective_func: String
                  ): java.util.List[Double] = {

    val distance = (distanceMin, distanceMax)
    val streamTrainSource = StreamFactory.getDomainStreamSource(TrainPath, domain, List.empty)
    objective_func match {
      case "comb" =>
      {
        val training_time = config(
          Key.exec.benchRuns -> 5,
          Key.verbose -> true,
          Key.exec.outliers.suspectPercent -> 20,
          Key.exec.outliers.covMultiplier -> 1.1
        ) withWarmer {
          new Warmer.Default
        } withMeasurer {
          new Measurer.Default
        } measure {
          val sdfap = SDFAProvider(SDFASourceFromSRE(PatternPath, ConfigUtils.defaultPolicy, DeclarationsPath))
          val spstp1 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource, pMin = pMin, alpha = alpha, gamma = gamma, r = r))
          val spstp = SPSTProvider(SPSTSourceDirectI(List(spstp1.provide().head)))
          //val fsmp1 = FSMProvider(spstp)
          val wtp0 = WtProvider(WtSourceSPST(
            spstp,
            horizon = horizon,
            cutoffThreshold = ConfigUtils.wtCutoffThreshold,
            distance = distance
          ))
          val wtp1 = WtProvider(WtSourceDirect(List(wtp0.provide().head)))
          //          val pp1 = ForecasterProvider(ForecasterSourceBuild(
          //            fsmp1,
          //            wtp1,
          //            horizon = horizon,
          //            confidenceThreshold = 0.5, //TODO CHECK
          //            maxSpread = maxSpread,
          //            method = method
          //          ))
        }
        val tt = training_time.value

        val sdfap = SDFAProvider(SDFASourceFromSRE(PatternPath, ConfigUtils.defaultPolicy, DeclarationsPath))
        val spstp1 = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource, pMin = pMin, alpha = alpha, gamma = gamma, r = r))
        SerializationUtils.write2File[SPSTInterface](spstp1.provide(), SavePath)
        val metrics = List(tt)
        metrics.asJava
      }
      case "nt" => {
        val streamTrainSource = StreamFactory.getDomainStreamSource(TrainPath, domain, List.empty)
        // read pattern file and declarations file & create a sdfa
        val sdfap = SDFAProvider(SDFASourceFromSRE(PatternPath, ConfigUtils.defaultPolicy, DeclarationsPath))
        // create a spst with the following parameters and via the use of the stream source
        val spstps = SPSTProvider(SPSTSourceFromSDFA(sdfap, order, streamTrainSource,
          pMin = pMin, alpha = alpha, gamma = gamma, r = r))

        val wtp = WtProvider(
          WtSourceSPST(
            spstps,
            horizon = horizon,
            cutoffThreshold = ConfigUtils.wtCutoffThreshold,
            distance = distance
          )
        )

        SerializationUtils.write2File[SPSTInterface](spstps.provide(), SavePath)
        SerializationUtils.write2File[Map[Int, WtDistribution]](wtp.provide(), SavePath+".wtd")

        val metrics = List(order.toDouble)
        metrics.asJava
      }
      case _ => throw new IllegalArgumentException("Stat not recognized: " + objective_func)
    }
  }

  def loadAndTest(testPath: String,
                  loadPath: String,
                  domain: String,
                  distanceMin: Double,
                  distanceMax: Double,
                  runConfidenceThreshold: Double,
                 ):java.util.List[Double] = {

    val distance = (distanceMin, distanceMax)

    val fsmp = FSMProvider(SPSTProvider(SPSTSourceSerialized(loadPath)))
    val wtdFileExists = Files.exists(Paths.get(loadPath+".wtd"))
    val wtp = wtdFileExists match {
      case true => WtProvider(WtSourceSerialized(loadPath+".wtd"))
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
        confidenceThreshold = runConfidenceThreshold,
        maxSpread = maxSpread,
        method = method
      )
    )

    val streamTestSource = StreamFactory.getDomainStreamSource(testPath, domain, List.empty)

    val erft = ERFTask(
      fsmp = fsmp,
      pp = pp,
      predictorEnabled = true,
      finalsEnabled = false,
      expirationDeadline = ConfigUtils.defaultExpiration,
      distance = distance,
      streamSource = streamTestSource
    )

    val prof = erft.execute()

    val resultsList = prof.getTPTNFPFN(0)
    resultsList.asJava
  }

  def runOOFEngine(loadPath: String,
                   distanceMin: Double,
                   distanceMax: Double,
                   runConfidenceThreshold: Double,
                   inputStreamTopic: String,
                   syncTopic: String,
                   modelsTopic: String,
                   wSyncTopic: String,
                   statsTopic: String,
                   resultsTopic: String,
                   schemasPath: String,
                   statsReportingDistance: Long,
                   startTime: Long,
                   domain: String
                  ): Unit = {

    // -------------- KAFKA STUFF ---------------
    println("Starting OOF engine...")

    // --- SCHEMAS
    // paths
    var keyPath = schemasPath + "/" + inputStreamTopic + "-key.avsc"
    var valuePath = schemasPath + "/" + inputStreamTopic + "-value.avsc"
    val inputStreamSchemasPaths = Map[String,String]("key" -> keyPath, "value"-> valuePath)

    keyPath = schemasPath + "/" + syncTopic + "-key.avsc"
    valuePath = schemasPath + "/" + syncTopic + "-value.avsc"
    val syncTopicSchemasPaths = Map[String,String]("key" -> keyPath, "value" -> valuePath)

    keyPath = schemasPath + "/" + modelsTopic + "-key.avsc"
    valuePath = schemasPath + "/" + modelsTopic + "-value.avsc"
    val modelsTopicSchemasPaths = Map[String, String]("key"-> keyPath,"value"-> valuePath)

    keyPath = schemasPath + "/" + wSyncTopic + "-key.avsc"
    valuePath = schemasPath + "/" + wSyncTopic + "-value.avsc"
    val wSyncTopicSchemasPaths = Map[String,String]("key"->keyPath, "value" -> valuePath)

    keyPath = schemasPath + "/" + wSyncTopic + "-key.avsc"
    valuePath = schemasPath + "/" + wSyncTopic + "-value.avsc"
    val resultsTopicSchemasPaths = Map[String, String]("key" -> keyPath, "value" -> valuePath)

    keyPath = schemasPath + "/" + statsTopic + "-key.avsc"
    valuePath = schemasPath + "/" + statsTopic + "-value.avsc"
    val statsTopicSchemasPaths = Map[String, String]("key" -> keyPath, "value" -> valuePath)

    // PRODUCER PROPS
    val producerProps:Properties = new Properties()
    producerProps.put("bootstrap.servers", "localhost:9092")
    producerProps.put("acks","all")

    // CONSUMER PROPS
    val consumerProps:Properties = new Properties()
    consumerProps.put("bootstrap.servers", "localhost:9092")
    consumerProps.put("enable.auto.commit", "true")
    consumerProps.put("auto.commit.interval.ms", "1000")
    consumerProps.put("auto.offset.reset", "earliest")
    consumerProps.put("max.poll.records", "10000");

    // Input Stream
    val streamConsumerProperties = consumerProps.clone().asInstanceOf[Properties]
    streamConsumerProperties.put("group.id", (java.util.UUID).randomUUID().toString())
    // Models
    val modelConsumerProperties = consumerProps.clone().asInstanceOf[Properties]
    modelConsumerProperties.put("group.id",
      (java.util.UUID).randomUUID().toString())
    // Sync
    val syncConsumerProperties = consumerProps.clone().asInstanceOf[Properties]
    syncConsumerProperties.put("group.id",
      (java.util.UUID).randomUUID().toString())
    // wSync
    val wSyncProducerProperties = producerProps.clone().asInstanceOf[Properties]
    // results
    val resultsProducerProperties = producerProps.clone().asInstanceOf[Properties]

    // results
    val statsProducerProperties = producerProps.clone().asInstanceOf[Properties]
    // Consumers
    val streamConsumer: AvroConsumer = AvroConsumer(streamConsumerProperties, inputStreamSchemasPaths, inputStreamTopic)
    val modelConsumer: AvroConsumer = AvroConsumer(modelConsumerProperties, modelsTopicSchemasPaths, modelsTopic)
    val syncConsumer: AvroConsumer = AvroConsumer(syncConsumerProperties, syncTopicSchemasPaths, syncTopic)

    // Producers
    val wSyncProducer: AvroProducer = AvroProducer(wSyncProducerProperties, wSyncTopicSchemasPaths, wSyncTopic)
    val resultsProducer: AvroProducer = AvroProducer(resultsProducerProperties, resultsTopicSchemasPaths, resultsTopic)
    val statsProducer: AvroProducer = AvroProducer(statsProducerProperties, statsTopicSchemasPaths, statsTopic )


    // --------------- WAYEB STUFF --------------------
    val distance = (distanceMin, distanceMax)

    val fsmp = FSMProvider(SPSTProvider(SPSTSourceSerialized(loadPath)))
    val wtdFileExists = Files.exists(Paths.get(loadPath+".wtd"))

    val wtp = wtdFileExists match {
      case true => WtProvider(WtSourceSerialized(loadPath+".wtd"))
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
        confidenceThreshold = runConfidenceThreshold,
        maxSpread = maxSpread,
        method = method
      )
    )

      println(domain)
      val streamSource = domain match {
        case "maritimeOOF" => MaritimeWAStreamSourceOOF(streamConsumer, modelConsumer, syncConsumer,
          wSyncProducer, startTime)
        case "cardsOOF" => TrxStreamSourceOOF(streamConsumer, modelConsumer, syncConsumer,
          wSyncProducer, startTime)
        case _ => null
      }
      val erft = OOFERFTask(
        fsmp = fsmp,
        pp = pp,
        predictorEnabled = true,
        finalsEnabled = false,
        expirationDeadline = ConfigUtils.defaultExpiration,
        distance = distance, horizon = horizon, maxSpread = maxSpread,
        method = method,
        streamSource = streamSource,
        outputStreamProducer = resultsProducer,
        statsProducer = statsProducer,
        statsReportingDistance = statsReportingDistance
    )
    erft.execute()
  }
}

