package ui.experiments.archived.caviar

object Constants {
  val dataDir: String = System.getenv("CAVIARDIR")
  val wayebHome: String = System.getenv("WAYEB_HOME")
  val resultsDir: String = dataDir + "/results"
  val sampling: Int = 5
}
