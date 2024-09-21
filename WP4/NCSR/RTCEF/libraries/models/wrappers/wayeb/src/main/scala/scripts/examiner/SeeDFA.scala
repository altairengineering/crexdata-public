package scripts.examiner

import workflow.provider.{FSMProvider, SDFAProvider, SPSAProvider}
import workflow.provider.source.sdfa.SDFASourceSerialized
import workflow.provider.source.spsa.SPSASourceSerialized

object SeeDFA {
  def main(args: Array[String]): Unit = {
    val dfaFile = args(0)
    val dfaType = args(1)
    val dfap = dfaType match {
      case "symbolic" => FSMProvider(SDFAProvider(new SDFASourceSerialized(dfaFile)))
      case "spsa" => FSMProvider(SPSAProvider(new SPSASourceSerialized(dfaFile)))
    }
    val sdfa = dfap.provide()
    println(sdfa.size + " FSMs found")
    println(sdfa.foldLeft("") { (acc, x) => acc + "\t\t\tSDFA:\n" + x.size })
    println(sdfa.head)
  }

}
