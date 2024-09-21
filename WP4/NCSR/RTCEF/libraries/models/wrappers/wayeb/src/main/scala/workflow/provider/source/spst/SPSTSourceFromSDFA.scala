package workflow.provider.source.spst

import stream.source.StreamSource
import workflow.provider.SDFAProvider

object SPSTSourceFromSDFA {
  def apply(
             sdfap: SDFAProvider,
             order: Int,
             trainStreamSource: StreamSource,
             pMin: Double,
             alpha: Double,
             gamma: Double,
             r: Double
           ): SPSTSourceFromSDFA = new SPSTSourceFromSDFA(sdfap, order, trainStreamSource, pMin, alpha, gamma, r)
}

class SPSTSourceFromSDFA(
                          val sdfap: SDFAProvider,
                          val order: Int,
                          val trainStreamSource: StreamSource,
                          val pMin: Double,
                          val alpha: Double,
                          val gamma: Double,
                          val r: Double
                        ) extends SPSTSource {

}
