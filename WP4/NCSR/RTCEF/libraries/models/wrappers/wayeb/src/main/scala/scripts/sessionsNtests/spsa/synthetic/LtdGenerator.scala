package scripts.sessionsNtests.spsa.synthetic

import breeze.stats.distributions.Uniform

object LtdGenerator {
  def apply(
      order: Int,
      prob: Double,
      control: String,
      target: String,
      middle: String
  ): LtdGenerator = new LtdGenerator(order, prob, control, target, middle)

  def apply(
      order: Int,
      prob: Double
  ): LtdGenerator = new LtdGenerator(order, prob, "0", "1", order.toString)
}

class LtdGenerator(
    order: Int,
    prob: Double,
    control: String,
    target: String,
    middle: String
) {
  require(order >= 2)

  private val ltd1 = generateLtd(middle, control)
  private val ltd2 = generateLtd(control, target)
  private val uni = Uniform(0, 2)

  def sampleLtd: List[String] = {
    val s = uni.sample()
    if (s < prob) ltd1 else ltd2
  }

  def generateLtd(
      first: String,
      last: String
  ): List[String] = {
    generateLtdAux(last, List(first))
  }

  def generateLtdAux(
      last: String,
      ltd: List[String]
  ): List[String] = {
    if (ltd.length == order + 1) ltd
    else if (ltd.length == order) ltd ::: List(last)
    else generateLtdAux(last, ltd ::: List(middle))
  }

}
