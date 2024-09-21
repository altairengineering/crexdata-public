package scripts.data.bio

object Constants {
  final val home = System.getenv("WAYEB_HOME")
  final val dataDir = home + "/scripts/data/bio/symbolic_sims_12_symbols"
  final val pathToInterestingDir = dataDir + "/interesting"
  final val pathToNotInterestingDir = dataDir + "/not_interesting"
  final val foldsNo: Int = 5
  final val percentTrain = 1.0 - (1 / foldsNo.toDouble)
}
