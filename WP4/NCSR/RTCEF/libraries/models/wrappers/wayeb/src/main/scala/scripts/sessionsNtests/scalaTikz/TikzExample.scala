package scripts.sessionsNtests.scalaTikz

import math._
import scalatikz.pgf.plots.Figure
import scalatikz.pgf.plots.enums.LegendPos.SOUTH_WEST
import scalatikz.pgf.enums.LineStyle.DASHED

object TikzExample extends App {

  val domain = BigDecimal(-2 * Pi) to BigDecimal(2 * Pi) by 0.1

  Figure("sin_vs_cosine")
    .plot(domain -> sin _)
    .plot(lineStyle = DASHED)(domain -> cos _)
    .havingLegends("$\\sin(x)$", "$\\cos(x)$")
    .havingLegendPos(SOUTH_WEST)
    .havingXLabel("$X$")
    .havingYLabel("$Y$")
    .havingTitle("$\\sin(x)$ vs $\\cos(x)$")
    .show()

}
