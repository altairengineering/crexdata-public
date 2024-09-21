package workflow.provider.source.wt

object WtSourceSerialized {
  def apply(fn: String): WtSourceSerialized = new WtSourceSerialized(fn)
}

class WtSourceSerialized(val fn: String) extends WtSource {

}
