package scripts.data.cards

object Constants {
  val wayebHome: String = System.getenv("WAYEB_HOME")
  val dataDir: String = wayebHome + "/data/cards"
  val numEvents: Int = 1000000
  val foldsNo: Int = 4
  val generatedFile: String = dataDir + "/generated/" + numEvents + ".csv"
  val withReset: Boolean = false
  val patternFilePath: (String, String, String) =
    //(wayebHome + "/patterns/cards/flash/pattern.sre", "nonoverlap", "flash")
    //(wayebHome + "/patterns/cards/flash/patternGT.sre", "nonoverlap", "flashGT")
    (wayebHome + "/patterns/cards/increasing/pattern.sre", "nonoverlap", "increasing")
    //(wayebHome + "/patterns/cards/increasing/patternGT.sre", "nonoverlap", "increasingGT")
    //(wayebHome + "/patterns/cards/decreasing/pattern.sre", "nonoverlap", "decreasing")
    //(wayebHome + "/patterns/cards/decreasing/patternGT.sre", "nonoverlap", "decreasingGT")
    //(wayebHome + "/patterns/cards/increasing/patternPaths.sre", "nonoverlap", "increasingPaths")
}
