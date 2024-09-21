package scripts.examiner

import workflow.provider.PSAProvider
import workflow.provider.source.psa.PSASourceSerialized

object SeePSA {
  def main(args: Array[String]): Unit = {
    val psaFile = args(0)
    val psap = PSAProvider(new PSASourceSerialized(psaFile))
    val psa = psap.provide().head
    println("PSA generated\n " + psa.toString)
    println("PSA size/maxOrder/ordersCounts: " + psa.size + "/" + psa.maxOrder + "/" + psa.statesByOrder)
  }

}
