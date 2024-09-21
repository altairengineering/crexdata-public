package scripts.sessionsNtests.spsa.synthetic

import breeze.stats.distributions.Uniform

class LtdGenerator1(
    order: Int,
    symbols: Set[String],
    target: String,
    dummyOrTargetProb: Double
) {
  require(order >= 2)
  require(symbols.contains(target))
  require(dummyOrTargetProb > 0.0 & dummyOrTargetProb < 1.0)

  private val symbolsNotTarget = symbols - target
  require(order <= symbolsNotTarget.size)
  private val dummyOrTargetDist = Uniform(0, 1)
  private val dummyDist = Uniform(0, symbolsNotTarget.size)
  private val dummyMap = symbolsNotTarget.zipWithIndex.map(e => (e._2, e._1)).toMap
  private val targetLtd = symbolsNotTarget.toList.take(order) ::: List(target)

  def sampleLtd: List[String] = {
    val s = dummyOrTargetDist.sample()
    if (s < dummyOrTargetProb) sampleDummy else sampleTarget
  }

  private def sampleDummy: List[String] = {
    //List(dummyMap(dummyDist.sample.toInt),dummyMap(dummyDist.sample.toInt))
    //List(targetLtd.drop(order-1).head,dummyMap(dummyDist.sample.toInt))
    val newDummy = sampleDummies(order, List.empty[String], targetLtd.take(order))
    newDummy
  }

  private def sampleDummies(
      length: Int,
      current: List[String],
      remaining: List[String]
  ): List[String] = {
    require(length >= 0)
    if (length == 0) current
    else {
      val newStart = remaining.head
      val newDummy = sampleDummyOfLength(length + 1, List(newStart))
      sampleDummies(length - 1, current ::: newDummy, remaining.tail)
    }
  }

  private def sampleDummyOfLength(
      length: Int,
      current: List[String]
  ): List[String] = {
    if (current.length == length) current
    else sampleDummyOfLength(length, current ::: List(dummyMap(dummyDist.sample.toInt)))
  }

  private def sampleTarget: List[String] = {
    targetLtd
  }

}
