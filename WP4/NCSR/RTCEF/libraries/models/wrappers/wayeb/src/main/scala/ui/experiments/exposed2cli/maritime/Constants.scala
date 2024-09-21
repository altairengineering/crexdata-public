package ui.experiments.exposed2cli.maritime

object Constants {
  val wayebHome: String = System.getenv("WAYEB_HOME")
  val dataDir: String = wayebHome + "/data/maritime"
  val resultsDir: String = wayebHome + "/results/maritime"
}
