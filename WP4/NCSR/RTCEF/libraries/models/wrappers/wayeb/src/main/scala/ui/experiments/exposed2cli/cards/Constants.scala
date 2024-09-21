package ui.experiments.exposed2cli.cards

object Constants {
  val wayebHome: String = System.getenv("WAYEB_HOME")
  val dataDir: String = wayebHome + "/data/cards"
  val resultsDir: String = wayebHome + "/results/cards"
}
