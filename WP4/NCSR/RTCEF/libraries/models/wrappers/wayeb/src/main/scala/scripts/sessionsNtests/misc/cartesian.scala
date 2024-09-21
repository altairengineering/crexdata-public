package scripts.sessionsNtests.misc

import utils.SetUtils

object cartesian {

  def main(args: Array[String]): Unit = {
    val set1 = Set(1, 2)
    val set2 = Set(3, 4)
    val set3 = Set(6, 7)

    val cart = SetUtils.cartesian(Set(set1, set2, set3))

    println(cart)
  }

}
