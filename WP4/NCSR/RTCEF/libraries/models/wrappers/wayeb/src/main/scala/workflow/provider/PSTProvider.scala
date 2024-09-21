package workflow.provider

import fsm.SDFAInterface
import stream.source.StreamSource
import model.vmm.{Isomorphism, VMMUtils}
import model.vmm.pst.PredictionSuffixTree
import workflow.provider.source.pst.{PSTSource, PSTSourceDirect, PSTSourceLearner}
import workflow.condition.Condition

object PSTProvider {
  /**
    * Constructor for PST provider.
    *
    * @param pstSource The source for the PST:
    *                  - PSTSourceDirect when the PST already exists.
    *                  - PSTSourceLearner when the PST must be learnt.
    * @return A PST provider.
    */
  def apply(pstSource: PSTSource): PSTProvider = new PSTProvider(pstSource, List.empty)
}

/**
  * According to type of source for PST, builds a provider in the form of a list of PSTs along with their respective
  * isomorphism.
  *
  * @param pstSource The source for the PST:
  *                  - PSTSourceDirect when the PST already exists.
  *                  - PSTSourceLearner when the PST must be learnt.
  * @param conditions A list of conditions that must be checked and satisfied.
  */
class PSTProvider private (
                            pstSource: PSTSource,
                            conditions: List[Condition]
                          ) extends AbstractProvider(conditions) {

  checkConditions()

  /**
    * Calling this function actually initiates the construction of the PSTs.
    * Before calling this, nothing is done.
    * CAUTION: Do not create a provider, then delete its source (e.g. the file the the serialized PST) and then call
    * provide(). Keep the source until you call provide().
    *
    * @return A list of PSTs with their isomorphisms.
    */
  override def provide(): List[(PredictionSuffixTree, Isomorphism)] = {
    pstSource match {
      case x: PSTSourceDirect => x.pst
      case x: PSTSourceLearner => learnPSTs(x.sdfap, x.trainStream, x.maxOrder, x.pMin, x.alpha, x.gammaMin, x.r)
      case _ => throw new Error("Not valid PSTSource")
    }

  }

  /**
    * Learns PSTs from the given SDFA and a training stream.
    *
    * @param sdfap The provider for the SDFA.
    * @param streamSource The source for the training stream (same for all PSTs).
    * @param maxOrder The maximum order of the PST (same for all PSTs).
    * @param pMin This is the symbol threshold. Symbols with lower probability are discarded (same for all PSTs).
    * @param alpha Used to calculate the conditional threshold = (1 + alpha) * gammaMin. The conditional on the expanded
    *              context must be greater than this threshold (same for all PSTs).
    * @param gammaMin Used to calculate the conditional threshold = (1 + alpha) * gammaMin. The conditional on the
    *                 expanded context must be greater than this threshold (same for all PSTs).
    * @param r This is the likelihood ratio threshold. Contexts are expanded if the probability ratio of the conditional
    *          on the expanded context by the conditional on the original context is greater than this threshold (same
    *          for all PSTs).
    * @return A list of prediction suffix trees and their isomorphisms.
    */
  private def learnPSTs(
                         sdfap: SDFAProvider,
                         streamSource: StreamSource,
                         maxOrder: Int,
                         pMin: Double,
                         alpha: Double,
                         gammaMin: Double,
                         r: Double
                       ): List[(PredictionSuffixTree, Isomorphism)] = {
    val sdfais = sdfap.provide()
    val pstWithIsos = sdfais.map(sdfai => learnPST(sdfai, streamSource, maxOrder, pMin, alpha, gammaMin, r))
    pstWithIsos
  }

  /**
    * Learns a prediction suffix tree from a SDFA and a training stream.
    *
    * @param sdfai The SDFA interface,
    * @param streamSource The source for the training stream.
    * @param maxOrder The maximum order of the PST.
    * @param pMin This is the symbol threshold. Symbols with lower probability are discarded.
    * @param alpha Used to calculate the conditional threshold = (1 + alpha) * gammaMin. The conditional on the expanded
    *              context must be greater than this threshold.
    * @param gammaMin Used to calculate the conditional threshold = (1 + alpha) * gammaMin. The conditional on the
    *                 expanded context must be greater than this threshold.
    * @param r This is the likelihood ratio threshold. Contexts are expanded if the probability ratio of the conditional
    *          on the expanded context by the conditional on the original context is greater than this threshold.
    * @return A prediction suffix tree and its isomorphism.
    */
  private def learnPST(
                        sdfai: SDFAInterface,
                        streamSource: StreamSource,
                        maxOrder: Int,
                        pMin: Double,
                        alpha: Double,
                        gammaMin: Double,
                        r: Double
                      ): (PredictionSuffixTree, Isomorphism) = {
    val sdfa = sdfai.sdfa
    val partitionAttribute = sdfai.partitionAttribute
    val (pst, iso) = VMMUtils.learnPredictionSuffixTree(
      streamSource,
      sdfa,
      partitionAttribute,
      maxOrder,
      pMin,
      alpha,
      gammaMin,
      r
    )
    (pst, iso)
  }

  /**
    * Checks all conditions.
    * TODO: do the checking in the super-class workflow.provider.AbstractProvider?
    *
    * @return True if all conditions satisfied.
    */
  def checkConditions(): Boolean = {
    val superchecks = super.check()
    if (superchecks.contains(false)) {
      throw new Error("Provider conditions unsatisfied")
      false
    } else true
  }

}
