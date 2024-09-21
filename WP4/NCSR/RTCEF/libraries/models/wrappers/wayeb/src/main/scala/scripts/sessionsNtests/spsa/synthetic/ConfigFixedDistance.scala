package scripts.sessionsNtests.spsa.synthetic

import model.waitingTime.ForecastMethod.ForecastMethod
import model.waitingTime.ForecastMethod

object ConfigFixedDistance {
  val home: String = System.getenv("WAYEB_HOME")
  val resultsDir: String = home + "/results/psa"
  val resultsFile: String = resultsDir + "/results.csv"
  val testsNo: Int = 10
  val horizon: Int = 200
  val maxSpread: List[Int] = List(1) //List(0,1,2,3,4,5,6,7,8,9)
  val predictionThreshold: List[Double] = List(0.2) //List(0.001)
  val spreadMethod: ForecastMethod = ForecastMethod.SMARTSCAN //"fixed-spread"
  val distance: List[Int] = List(-1) //List(1,2,3,4,5)
  val checkForEmitting = false // true
  val estimatedDistance: Int = 2
  val symbolsNo: Int = 20
  val disOrder: Int = 2
  val spsaOrder: Int = 5
  val trainEventsNo: Int = 1000000
  val testEventsNo: Int = 10000
  val target = "1"
  val whichLtdProb: Double = 0.2
  val symbolOrLtdProb: Double = 0.8
  val maxNoStates: Int = 500
  val expansionProb: Double = 0.8
  val finalsEnabled: Boolean = false
}
