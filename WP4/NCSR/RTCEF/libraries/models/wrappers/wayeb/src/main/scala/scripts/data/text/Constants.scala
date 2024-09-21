package scripts.data.text

object Constants {
  val wayebHome: String = System.getenv("WAYEB_HOME")
  val dataDir: String = wayebHome + "/scripts/data/compression/large"
  val outDir: String = System.getenv("TEXTDIR")
  val txtFile: String = dataDir + "/bible.txt"
  val streamFile: String = outDir + "/stream/bible.csv"
  val foldsNo: Int = 10

}
