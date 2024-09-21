package ui.experiments.exposed2cli

import workflow.provider.FSMProvider

object Utils {

  def findDistances(fsmp: FSMProvider): List[(Double, Double)] = {
    val fsmi = fsmp.provide().head
    val shortestPaths = fsmi.findShortestPathDistances
    val maxPath = shortestPaths.values.max
    val percentages = (1 to maxPath).map(x => x.toDouble / maxPath).toList
    val intervals = percentages.map(x => percent2interval(x))
    intervals
  }

  private def percent2interval(p: Double): (Double, Double) = {
    require(p >= 0.0 & p <= 1.0)
    val left = math.floor(p * 100) / 100
    val right = math.ceil(p * 100) / 100
    (left, right)
  }

}
