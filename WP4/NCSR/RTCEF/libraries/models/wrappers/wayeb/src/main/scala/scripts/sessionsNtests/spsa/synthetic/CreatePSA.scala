package scripts.sessionsNtests.spsa.synthetic

import utils.SerializationUtils
import model.vmm.Symbol
import model.vmm.pst.psa.{PSAUtils, ProbSuffixAutomaton}

object CreatePSA {

  def main(args: Array[String]): Unit = {
    val home = System.getenv("WAYEB_HOME")
    val psaFn = home + "/results/psaOrder10Symbols3.psa"
    val maxOrder = 1
    val symbolsNo = 2
    val expansionProb = 0.7
    val symbols = (1 to symbolsNo).toSet[Int].map(i => Symbol(i))
    val psa = PSAUtils.createPSA(symbols, maxOrder, expansionProb)
    SerializationUtils.write2File[ProbSuffixAutomaton](List(psa), psaFn)
    println("PSA generated\n " + psa.toString)
    println("PSA size/maxOrder/ordersCounts: " + psa.size + "/" + psa.maxOrder + "/" + psa.statesByOrder)
  }

}
