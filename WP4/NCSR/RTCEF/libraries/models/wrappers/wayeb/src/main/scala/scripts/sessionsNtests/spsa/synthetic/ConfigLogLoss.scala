package scripts.sessionsNtests.spsa.synthetic

object ConfigLogLoss {
  val home: String = System.getenv("WAYEB_HOME")
  val resultsDir: String = home + "/results/psa"
  val resultsFile: String = resultsDir + "/results.csv"
  val testsNo: Int = 1
  val symbolsNo: Int = 50
  val disOrder: Int = 1
  val spsaOrder: Int = 2
  val trainEventsNo: Int = 1000000
  val testEventsNo: Int = 10000
  val target = "0"
  val whichLtdProb: Double = 0.9
  val symbolOrLtdProb: Double = 0.5
  val maxNoStates: Int = 500
  val expansionProb: Double = 0.8
}
