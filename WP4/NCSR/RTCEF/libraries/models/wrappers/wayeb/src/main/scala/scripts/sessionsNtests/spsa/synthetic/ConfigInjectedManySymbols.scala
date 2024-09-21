package scripts.sessionsNtests.spsa.synthetic

object ConfigInjectedManySymbols {
  val home: String = System.getenv("WAYEB_HOME")
  val resultsDir: String = home + "/results/psa"
  val resultsFile: String = resultsDir + "/results.csv"
  val testsNo: Int = 1
  val maxOrder: Int = 2
  val disMaxOrder: Int = 1
  val symbolsNo: Int = 100
  val trainEventsNo: Int = 1000000
  val testEventsNo: Int = 10000
  val horizon: Int = 200
  val maxSpread: List[Int] = List(100, 90, 80, 60, 50, 40, 30, 20, 10, 5, 1)
  val predictionThreshold: List[Double] = List(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  val maxNoStates: Int = 1000
  val whichLtdProb: Double = 0.9
  val symbolOrLtdProb: Double = 0.5
  val target = "1"
  val finalsEnabled: Boolean = false

  override def toString: String =
    "\nHOME: " + home +
      "\nresultsDir: " + resultsDir +
      "\nresultsFile: " + resultsFile +
      "\ntestsNo: " + testsNo +
      "\nmaxOrder: " + maxOrder +
      "\ndisMaxOrder: " + disMaxOrder +
      "\nsymbolsNo: " + symbolsNo +
      "\ntrainEventsNo: " + trainEventsNo +
      "\ntestEventsNo: " + testEventsNo +
      "\nhorizon: " + horizon +
      "\nmaxSpread: " + maxSpread +
      "\npredictionThreshold: " + predictionThreshold +
      "\nmaxNoStates: " + maxNoStates +
      "\nwhichLtdProb: " + whichLtdProb +
      "\nsymbolOrLtdProb: " + symbolOrLtdProb +
      "\ntarget: " + 1
}
