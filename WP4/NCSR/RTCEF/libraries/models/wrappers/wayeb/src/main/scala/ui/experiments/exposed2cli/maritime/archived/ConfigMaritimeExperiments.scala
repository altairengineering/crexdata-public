package ui.experiments.exposed2cli.maritime.archived

object ConfigMaritimeExperiments {
  val dataDir: String = System.getenv("DATADIR")
  val wayebHome: String = System.getenv("WAYEB_HOME")
  val intervalSec: Int = 60 //10 * 60
  val maxGap: Int = 30 * 60
  val speedThreshold: Double = 1.0
  val startTime: Long = 1443650401
  val endTime: Long = 1459461585 //1446242401 // 1459461585 //startTime + 30*(24*60*60)
  val resultsDir: String = dataDir + "/results"
  val finalsEnabled: Boolean = false
}
