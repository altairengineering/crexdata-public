package scripts.data.synthetic

import model.vmm.pst.psa.PSAUtils
import model.vmm.{Symbol, SymbolWord}

object PSAStreamGenerator {
  def generate(): Unit = {
    val sym1 = Symbol(1)
    val sym0 = Symbol(0)
    val label1 = SymbolWord(List(sym0, sym0, sym1))
    val label2 = SymbolWord(List(sym0, sym0, sym0))
    val prob1For0 = 0.1
    val prob2For0 = 0.9
    val targetProb = 0.5
    val psa = PSAUtils.createPSA01(label1, prob1For0, label2, prob2For0, targetProb)
    val symbols2types: Map[Symbol, String] = Map(sym0 -> "A", sym1 -> "B")
    val (trainStream, _) = psa.generateStream(Config.trainSize, symbols2types)
    val (testStream, _) = psa.generateStream(Config.testSize, symbols2types)
    trainStream.writeCSV(Config.trainFile)
    testStream.writeCSV(Config.testFile)
  }

}
