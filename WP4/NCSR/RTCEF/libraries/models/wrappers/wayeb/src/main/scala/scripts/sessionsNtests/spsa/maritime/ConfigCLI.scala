package scripts.sessionsNtests.spsa.maritime

object ConfigCLI {
  val home: String = System.getenv("WAYEB_HOME")
  val dataDir: String = "/home/zmithereen/data/maritime"

  val finalsEnabled: Boolean = false
  val patternFile: String = home + "/patterns/datacron/maritime/anchor/pattern.sre"
  //val trainSet: String = dataDir + "/test.csv"
  //val testSet: String = dataDir + "/test.csv"
  val trainSet: String = dataDir + "/sync/ais_brest_sync_1443650401-1446242401_gap1800_interval60_speed0.0.csv"
  val testSet: String = dataDir + "/sync/ais_brest_synopses_sync_1443650401-1459461585_gap1800_interval60.csv"
  val domainSpecificStream: String = "maritime"
  val threshold = 0.5

  /*val finalsEnabled: Boolean = true
  val patternFile: String = home + "/patterns/datacron/maritime/collision/collisionFinals.sre"
  val trainSet: String = dataDir + "/ais_brest_sync_paired_100k_gap1800_interval60.csv" //"/228133000228854000.csv"
  val testSet: String = dataDir  + "/ais_brest_sync_paired_100k_gap1800_interval60.csv" //"/228133000228854000.csv"
  val threshold = 0.3*/

  val declarationsFile: String = home + "/patterns/datacron/maritime/anchor/testDeclarations.sre"
  val resultsDir: String = home + "/results/maritime/collision"
  val maxSpread = 100
  val horizon = 500
  val gap: Int = 1800

}
