package scripts.data.synthetic

object Config {
  val wayebHome: String = System.getenv("WAYEB_HOME")
  val psaDir: String = System.getenv("PSADIR")
  val trainFile: String = psaDir + "/generated/train.csv"
  val testFile: String = psaDir + "/generated/test.csv"
  val trainSize: Int = 1000
  val testSize: Int = 100
  val patternFile: String = wayebHome + "/patterns/psa/001.sre"
}
