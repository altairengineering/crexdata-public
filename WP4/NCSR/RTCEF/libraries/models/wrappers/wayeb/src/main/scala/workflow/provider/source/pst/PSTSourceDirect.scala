package workflow.provider.source.pst

import model.vmm.Isomorphism
import model.vmm.pst.PredictionSuffixTree

class PSTSourceDirect(val pst: List[(PredictionSuffixTree, Isomorphism)]) extends PSTSource {

}
