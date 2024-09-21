package scripts.sessionsNtests.spsa.synthetic

import stream.StreamFactory
import ui.ConfigUtils
import workflow.provider.source.sdfa.SDFASourceFromSRE
import workflow.provider.source.spsa.SPSASourceFromSRE
import workflow.provider.{FSMProvider, SDFAProvider, SPSAProvider}
import workflow.task.engineTask.ERFTask

object TestSPSA {

  def main(args: Array[String]): Unit = {
    val home = System.getenv("WAYEB_HOME")
    val patternsFile = home + "/patterns/misc/a_seq_b.sre"
    val declarationsFile = home + "/patterns/misc/declarations1.sre"
    val stream = StreamFactory.getStreamSource(100, scala.collection.mutable.Map[String, Double]("A" -> 0.8, "B" -> 0.2), 99)

    val sdfap = SDFAProvider(SDFASourceFromSRE(patternsFile, ConfigUtils.defaultPolicy, declarationsFile))
    val fsmp = FSMProvider(sdfap)
    val erft = ERFTask(fsmp, stream)
    erft.execute()

    val maxNoStates = 100
    val spsap = SPSAProvider(SPSASourceFromSRE(patternsFile, declarationsFile, stream, ConfigUtils.defaultPolicy, maxNoStates))
    val fsmp1 = FSMProvider(spsap)
    val erft1 = ERFTask(fsmp1, stream)
    erft1.execute()

  }

}
