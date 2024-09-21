package scripts.sessionsNtests.spsa.maritime

import model.waitingTime.ForecastMethod.ForecastMethod
import model.waitingTime.ForecastMethod

object ConfigHighSpeed {
  val home: String = System.getenv("WAYEB_HOME")
  val resultsDir: String = home + "/results/psa"
  val resultsFile: String = resultsDir + "/results.csv"
  //val patternFile: String = home + "/patterns/datacron/maritime/highSpeed/highSpeed.sre"
  val patternFile: String = home + "/patterns/datacron/maritime/approachingBrestPort/vmm/pattern.sre"
  val patternFileVMM: String = home + "/patterns/datacron/maritime/approachingBrestPort/vmm/patternVMM.sre"
  val declarationsFile: String = home + "/patterns/datacron/maritime/approachingBrestPort/vmm/declarationsDistance1.sre"
  val minTermMethod: String = "withoutsat"
  //val streamFile: String =  "/mnt/Warehouse/data/maritime/cyril/synopses/ais_brest_synopses_v0.8/csv/trajectories/60sec/227592820.csv"
  val streamFile: String = "/home/zmithereen/data/maritime/brest/trajectories/60sec/227592820.csv"
  val horizon: Int = 200
  val testsNo: Int = 10
  val maxSpread: Int = 5
  val minDistance: Int = 0
  val predictionThreshold: Double = 0.5
  val spreadMethod: ForecastMethod = ForecastMethod.SMARTSCAN // "fixed-spread"
  val maxNoStates: Int = 1000
  val finalsEnabled: Boolean = false
}
