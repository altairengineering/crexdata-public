package scripts.sessionsNtests.spsa.synthetic

object ConfigNext {
  val home: String = System.getenv("WAYEB_HOME")
  val resultsDir: String = home + "/results/psa"
  val resultsFile: String = resultsDir + "/results.csv"
  val testsNo: Int = 1
  val maxOrder: Int = 3
  val expansionProb: Double = 0.8
  val disMaxOrder: Int = 1
  val symbolsNo: Int = 5
  val trainEventsNo: Int = 1000000
  val testEventsNo: Int = 10000
  val horizon: Int = 200
  val maxSpread: Int = 10
  val predictionThreshold: Double = 0.8
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
      "\nexpansionProb: " + expansionProb +
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
