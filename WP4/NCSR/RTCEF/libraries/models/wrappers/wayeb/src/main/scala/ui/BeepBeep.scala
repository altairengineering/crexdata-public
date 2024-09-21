package ui

import fsm.{SDFAInterface, SPSTInterface}
import profiler.ProfilerInterface
import stream.source.StreamSource
import stream.StreamFactory
import ui.experiments.exposed2cli.cards.CardsPatternExperiments
import ui.experiments.exposed2cli.maritime.MaritimePatternExperiments
import utils.SerializationUtils
import workflow.provider._
import workflow.provider.source.matrix.MCSourceSerialized
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.source.sdfa.SDFASourceSerialized
import workflow.provider.source.spst.{SPSTSourceDirectI, SPSTSourceSerialized}
import workflow.provider.source.wt.{WtSourceMatrix, WtSourceSPST}
import workflow.task.engineTask.ERFTask
import workflow.task.estimatorTask.MatrixMLETask
import workflow.task.fsmTask.{SDFATask, SPSTTask}
import ui.demo.{RunOFO, RunOOFService}

object BeepBeep {

  private[ui] def runFSMDisambiguation(config: WayebConfig): Unit = {
    config.modelType match {
      case "fmm" => runSDFADisambiguation(config)
      case _ => throw new IllegalArgumentException
    }
  }

  private def runSDFADisambiguation(config: WayebConfig): Unit = {
    val sdfat = SDFATask(config.patterns, config.declarations, config.policy)
    val sdfa = sdfat.execute()
    SerializationUtils.write2File[SDFAInterface](sdfa, config.outputFsm)
  }

  private[ui] def runMatrixEstimation(config: WayebConfig): Unit = {
    val trainStream = getStreamSource(config.streamFile, config.domainSpecificStream, config.streamArgs)
    val fsmp = getFSMProviderSerialized(config.modelType, config.fsmFile)
    val met = MatrixMLETask(fsmp, trainStream)
    val mcs = met.execute()._1
    SerializationUtils.write2File(mcs, config.outputMc)
  }

  private[ui] def runForecasting(config: WayebConfig): Unit = {
    val fsmp = getFSMProviderSerialized(config.modelType, config.fsmFile)
    val pp = config.modelType match {
      case "vmm" => {
        val spsti = fsmp.provide().map(x => x.asInstanceOf[SPSTInterface])
        val spstp = SPSTProvider(SPSTSourceDirectI(spsti))
        val wtp = WtProvider(
          WtSourceSPST(
            spstp,
            horizon         = config.horizon,
            cutoffThreshold = ConfigUtils.wtCutoffThreshold,
            distance        = ConfigUtils.defaultDistance
          )
        )
        ForecasterProvider(
          ForecasterSourceBuild(
            fsmp,
            wtp,
            horizon             = config.horizon,
            confidenceThreshold = config.confidenceThreshold,
            maxSpread           = config.maxSpread,
            method              = config.spreadMethod
          )
        )
      }
      case "fmm" => {
        val mcp = MarkovChainProvider(MCSourceSerialized(config.mcFile))
        val wtdp = WtProvider(WtSourceMatrix(fsmp, mcp, config.horizon, ConfigUtils.defaultFinalsEnabled))
        ForecasterProvider(ForecasterSourceBuild(fsmp, wtdp, config.horizon, config.confidenceThreshold, config.maxSpread, config.spreadMethod))
      }
    }
    val stream = getStreamSource(config.streamFile, config.domainSpecificStream, config.streamArgs)
    val erf = ERFTask(fsmp, pp, stream, show = ConfigUtils.defaultShowMatchesForecasts)
    val profiler = erf.execute().asInstanceOf[ProfilerInterface]
    profiler.printProfileInfo(config.statsFile)
  }

  private[ui] def runRecognition(config: WayebConfig): Unit = {
    val fsmp = getFSMProviderSerialized(config.modelType, config.fsmFile)
    val stream = getStreamSource(config.streamFile, config.domainSpecificStream, config.streamArgs)
    val erf = ERFTask(fsmp, stream)
    val profiler = erf.execute()
    profiler.printProfileInfo(config.statsFile)
  }

  private[ui] def runLearnSPST(config: WayebConfig): Unit = {
    val streamSource = getStreamSource(config.streamFile, config.domainSpecificStream, config.streamArgs)
    val spstt = SPSTTask(config.patterns, config.declarations, config.policy, streamSource, config.pMin, config.alpha, config.gammaMin, config.r)
    val spst = spstt.execute()
    SerializationUtils.write2File[SPSTInterface](spst, config.outputSpst)
  }

  private def getFSMProviderSerialized(
                                        modelType: String,
                                        fsmFile: String
                                      ): FSMProvider = {
    val fap: AbstractProvider = modelType match {
      case "fmm" => SDFAProvider(SDFASourceSerialized(fsmFile))
      case "vmm" => SPSTProvider(SPSTSourceSerialized(fsmFile))
      case _ => throw new IllegalArgumentException
    }
    val fsmp = FSMProvider(fap)
    fsmp
  }

  private def getStreamSource(
                               fn: String,
                               domain: String,
                               args: String
                             ): StreamSource = {
    if (domain == "") StreamFactory.getCSVStreamSource(fn)
    else {
      val splitArgs = args.split(",").map(_.trim).toList
      StreamFactory.getDomainStreamSource(fn, domain, splitArgs)
    }
  }

  private[ui] def runExperiments(config: WayebConfig): Unit = {
    config.experimentsDomain match {
      case "cards" => CardsPatternExperiments.main(Array.empty)
      case "maritime" => MaritimePatternExperiments.main(Array.empty)
      case _ => throw new IllegalArgumentException("Unrecognized domain.")
    }
  }

  private[ui] def runServer(config: WayebConfig): Unit = {
    RunOFO.main()
  }

  private[ui] def runEngineService(config: WayebConfig): Unit = {
    RunOOFService.main(config)
  }



  /*private def runDFADisambiguation(config: WayebConfig): Unit = {
    val eventTypes = if (config.streamFile == "") Set.empty[String]
    else {
      val streamSource = getStreamSource(config.streamFile, config.domainSpecificStream, config.streamArgs)
      val stream = streamSource.emitEventsAndClose(EmitMode.BUFFER)
      stream.getEventTypes
    }
    val dfat = DFATask(config.patterns, config.policy, config.order, eventTypes)
    val dfas = dfat.execute()
    SerializationUtils.write2File[DFAInterface](dfas, config.outputFsm)
  }*/
}
