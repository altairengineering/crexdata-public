package workflow.provider.source.pst

import stream.source.StreamSource
import workflow.provider.SDFAProvider

object PSTSourceLearner {
  def apply(
             sdfap: SDFAProvider,
             trainStream: StreamSource,
             maxOrder: Int,
             pMin: Double,
             alpha: Double,
             gammaMin: Double,
             r: Double
           ): PSTSourceLearner = new PSTSourceLearner(sdfap, trainStream, maxOrder, pMin, alpha, gammaMin, r)
}

class PSTSourceLearner(
                        val sdfap: SDFAProvider,
                        val trainStream: StreamSource,
                        val maxOrder: Int,
                        val pMin: Double,
                        val alpha: Double,
                        val gammaMin: Double,
                        val r: Double
                      ) extends PSTSource {

}
