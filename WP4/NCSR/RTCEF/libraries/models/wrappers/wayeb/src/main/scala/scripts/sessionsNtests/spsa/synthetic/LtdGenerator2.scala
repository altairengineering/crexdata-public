package scripts.sessionsNtests.spsa.synthetic

import breeze.stats.distributions.Uniform

class LtdGenerator2(
    target: String,
    order: Int,
    allSymbols: Set[String],
    prob: Double
) {

  require(order >= 2)
  require(prob > 0 & prob < 1.0)
  private val symbolsNotTarget = allSymbols - target
  private val context = sampleWord(symbolsNotTarget, order, List.empty)
  private val subcontext = context.dropRight(1)
  private val tailAtOrder = context.takeRight(1).head
  private val symbolsNotTargetNotTail = symbolsNotTarget - tailAtOrder
  //private val ltdDist = Uniform(0,1.0)

  // target at head and context follows from left to right
  def sampleLtd: List[String] = {
    val ltdDist = Uniform(0, 1.0)
    val sample = ltdDist.sample()
    if (sample < prob) {
      target :: context
    } else {
      val notTargetSample = sampleSymbol(symbolsNotTarget)
      val notTargetNotTailSample = sampleSymbol(symbolsNotTargetNotTail)
      notTargetSample :: subcontext ::: List(notTargetNotTailSample)
    }
  }

  def sampleLtdWithDistance: List[(String, Int)] = {
    val ltd = sampleLtd
    val distances = (0 to order)
    ltd.zip(distances)
  }

  def sampleLtdWithTag(distance: Int): List[(String, Boolean)] = {
    require(distance > 0 & distance <= order)
    val ltd = sampleLtd
    val tags = false :: (1 to order).map(x => if (x == distance) true else false).toList
    ltd.zip(tags)
  }

  private def sampleWord(
      symbols: Set[String],
      length: Int,
      word: List[String]
  ): List[String] = {
    require(length > 0)
    if (word.length == length) word
    else sampleWord(symbols, length, sampleSymbol(symbols) :: word)
  }

  private def sampleSymbol(symbols: Set[String]): String = {
    val uni = Uniform(0, symbols.size - 0.001)
    val index = uni.sample().toInt
    val a = symbols.toArray
    a(index)
  }

}
