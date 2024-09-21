package scripts.sessionsNtests.spsa.maritime

import com.typesafe.scalalogging.LazyLogging
import stream.StreamFactory
import model.vmm.VMMUtils
import ui.ConfigUtils
import workflow.provider.source.matrix.{MCSourceMLE, MCSourceSPSA}
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.source.sdfa.SDFASourceDirect
import workflow.provider.source.spsa.SPSASourceDirect
import workflow.provider.source.wt.{WtSourceDirect, WtSourceMatrix}
import workflow.provider.{FSMProvider, ForecasterProvider, MarkovChainProvider, SDFAProvider, SPSAProvider, WtProvider}
import workflow.task.engineTask.ERFTask
import workflow.task.fsmTask.SDFATask

object RunHighSpeed extends LazyLogging {
  def main(args: Array[String]): Unit = {
    logger.info("Running high speed")
    logger.info("getting stream")
    val stream = StreamFactory.getDomainStreamSource(ConfigHighSpeed.streamFile, "datacronMaritime", List("60"))

    logger.info("sdfa provider")
    val sdfat = SDFATask(ConfigHighSpeed.patternFile, ConfigHighSpeed.declarationsFile)
    val sdfai = sdfat.execute()
    val sdfa = sdfai.map(s => s.sdfa)
    //logger.info(sdfa.head.toString)
    val fsmp = FSMProvider(SDFAProvider(SDFASourceDirect(sdfa)))
    logger.info("matrix provider")
    val mcp = MarkovChainProvider(MCSourceMLE(fsmp, stream))
    logger.info("wt provider")
    val wtp = WtProvider(WtSourceMatrix(fsmp, mcp, ConfigHighSpeed.horizon, ConfigHighSpeed.finalsEnabled))
    logger.info("predictor provider")
    val pp = ForecasterProvider(ForecasterSourceBuild(
      fsmp,
      wtp, ConfigHighSpeed.horizon,
      ConfigHighSpeed.predictionThreshold,
      ConfigHighSpeed.maxSpread,
      ConfigHighSpeed.spreadMethod
    ))
    logger.info("erf task")
    val erft = ERFTask(fsmp, pp, stream, show = true)
    val prof = erft.execute()
    prof.printProfileInfo()

    val spsa = VMMUtils.learnSymbolicPSA(ConfigHighSpeed.patternFileVMM, ConfigHighSpeed.declarationsFile, stream, ConfigUtils.defaultPolicy, ConfigHighSpeed.maxNoStates)
    val spsaProv = SPSAProvider(SPSASourceDirect(List(spsa.head._1)))
    val fsmProv = FSMProvider(spsaProv)
    val matrixProv = MarkovChainProvider(MCSourceSPSA(fsmProv))
    val wtdProvLearnt = WtProvider(WtSourceMatrix(fsmProv, matrixProv, ConfigHighSpeed.horizon, ConfigHighSpeed.finalsEnabled))
    val wtdsLearnt = wtdProvLearnt.provide()
    val ppLearnt = ForecasterProvider(ForecasterSourceBuild(
      fsmProv,
      WtProvider(WtSourceDirect(wtdsLearnt)),
      ConfigHighSpeed.horizon,
      ConfigHighSpeed.predictionThreshold,
      ConfigHighSpeed.maxSpread,
      ConfigHighSpeed.spreadMethod
    ))
    val erftLearnt = ERFTask(fsmProv, ppLearnt, stream, show = true)
    val profLearnt = erftLearnt.execute()
    profLearnt.printProfileInfo()
  }

}
