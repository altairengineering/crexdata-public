package scripts.sessionsNtests.spsa.synthetic

object ConfigInjected {
  val home: String = System.getenv("WAYEB_HOME")
  val resultsDir: String = home + "/results/psa"
  val resultsFile: String = resultsDir + "/resultsinjected.csv"
  val maxOrder: Int = 4
  val disMaxOrder: Set[Int] = Set(4) //Set(1,2,3,4)
  val symbolsNo: Int = maxOrder + 1
  val trainEventsNo: Int = 1000000
  val testEventsNo: Int = 10000
  val testsNo: Int = 100
  val horizon: Int = 200
  val maxSpread: Int = 5 //200
  val predictionThreshold: Double = 0.7
  val maxNoStates: Int = 250
  val whichLtdProb: Double = 0.8
  val symbolOrLtdProb: Double = 0.9
}
