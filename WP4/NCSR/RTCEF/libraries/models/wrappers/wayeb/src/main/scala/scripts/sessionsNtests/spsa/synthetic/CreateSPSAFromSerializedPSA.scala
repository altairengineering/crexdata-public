package scripts.sessionsNtests.spsa.synthetic

import fsm.SPSAInterface
import fsm.symbolic.sre.SREUtils
import fsm.symbolic.sfa.SFAUtils
import fsm.symbolic.sfa.logic.{PredicateConstructor, Sentence}
import fsm.symbolic.sfa.sdfa.SDFAUtils
import fsm.symbolic.sfa.snfa.SNFAUtils
import fsm.classical.pattern.regexp.OperatorType.CONCAT
import fsm.classical.pattern.regexp.{OperatorNode, SymbolNode}
import stream.GenericEvent
import utils.SerializationUtils
import model.vmm.{Isomorphism, Symbol, VMMUtils}
import ui.ConfigUtils
import workflow.provider.source.psa.PSASourceSerialized
import workflow.provider.source.spsa.SPSASourceDirect
import workflow.provider.{PSAProvider, SPSAProvider}

object CreateSPSAFromSerializedPSA {
  def main(args: Array[String]): Unit = {
    //val patterns = PatternGenerator.generateREPatterns(ConfigUtils.noOfPatterns,Set('1','2'),ConfigUtils.patternMaxDepth)
    val p1 = PredicateConstructor.getEventTypePred("A")
    val p2 = PredicateConstructor.getEventTypePred("B")
    val p3 = PredicateConstructor.getEventTypePred("C")
    //val p4 = PredicateConstructor.getEventTypePred("D")
    //val p5 = PredicateConstructor.getEventTypePred("E")
    //val pc = PredicateConstructor.getEventTypePred("3")
    val exclusives = Set(Set(p1, p2, p3)) //,p4,p5))
    val extras = Set.empty[Sentence]

    val home = System.getenv("WAYEB_HOME")
    val psaFn = home + "/results/psaOrder10Symbols3.psa"
    val psap = PSAProvider(new PSASourceSerialized(psaFn))
    val psa = psap.provide()

    val patternAC = OperatorNode(CONCAT, List(SymbolNode("A"), SymbolNode("C")))
    val patternCB = OperatorNode(CONCAT, List(SymbolNode("C"), SymbolNode("B")))
    val patternACCB = OperatorNode(CONCAT, List(patternAC, patternCB))

    val policy = ConfigUtils.defaultPolicy
    val formula = SREUtils.re2formula(patternACCB)
    val snfaStream = SNFAUtils.buildSNFAForStream(formula)
    val sdfa = SFAUtils.determinizeI(snfaStream, exclusives, extras)
    SDFAUtils.checkForDead(sdfa)
    val sdfap = SDFAUtils.setPolicy(sdfa, policy)
    val sentences = sdfap.getSentences
    val s1 = sentences.filter(s => s.evaluate(GenericEvent("A", 1))).head
    val s2 = sentences.filter(s => s.evaluate(GenericEvent("B", 1))).head
    val s3 = sentences.filter(s => s.evaluate(GenericEvent("C", 1))).head
    //val s4 = sentences.filter(s => s.evaluate(GenericEvent("D",1))).head
    //val s5 = sentences.filter(s => s.evaluate(GenericEvent("E",1))).head

    val iso = Isomorphism(List(s1, s2, s3), List(Symbol(1), Symbol(2), Symbol(3)))

    //val (spsa,something) = VMMUtils.mergeSingleSFAPSA((formula,0,"$"),exclusives,extras,policy,psa.head)
    val spsa = VMMUtils.embedPSAinSDFA(sdfap, psa.head, iso)

    val spsaFn = home + "/results/spsaACCB.spsa"
    val spsap = SPSAProvider(SPSASourceDirect(List(spsa)))
    val spsai = spsap.provide()
    SerializationUtils.write2File[SPSAInterface](spsai, spsaFn)
  }

}
