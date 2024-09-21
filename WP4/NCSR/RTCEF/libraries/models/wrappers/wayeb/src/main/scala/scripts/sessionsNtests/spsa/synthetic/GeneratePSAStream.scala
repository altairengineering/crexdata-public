package scripts.sessionsNtests.spsa.synthetic

import stream.StreamFactory
import workflow.provider.PSAProvider
import workflow.provider.source.psa.PSASourceSerialized

object GeneratePSAStream {
  def main(args: Array[String]): Unit = {
    val home = System.getenv("WAYEB_HOME")
    val psaFn = home + "/results/psaOrder10Symbols3.psa"
    val psaStreamFn = home + "/results/psa10.csv"
    val psap = PSAProvider(new PSASourceSerialized(psaFn))
    val psa = psap.provide().head
    val eventsNo = 10000
    val (stream, _) = StreamFactory.getStream(psa, eventsNo)
    stream.writeCSV(psaStreamFn)
  }

}
