package scripts.data.maritime

object Constants {
  val wayebHome: String = System.getenv("WAYEB_HOME")
  val dataDir: String = System.getenv("MARITIMEDIR")
  val baseFile: String = "ais_brest_synopses"
  val synopsesFilePath: String = dataDir + "/synopses/" + baseFile + ".csv"
  val intervalSec: Int = 1 * 60
  val maxGap: Int = 30 * 60
  val speedThreshold: Double = 1.0
  val startTime: Long = 1443650401
  val endTime: Long = 1459461585 //1446242401 // 1459461585 //startTime + 30*(24*60*60)
  //val endTime: Long =  startTime + 30*(24*60*60)
  val patternFilePath: (String, Int, String) =
    (wayebHome + "/patterns/maritime/port/pattern.sre", 60, "port")
  //(wayebHome + "/patterns/maritime/anchor/pattern2.sre",5,"anchor")
  //(wayebHome + "/patterns/maritime/highSpeed/pattern.sre", 5, "speed")
  val syncedSynopsesFilePath: String = dataDir + "/sync/" + baseFile + "_sync_" + startTime + "-" + endTime + "_gap" + maxGap + "_interval" + intervalSec + "_speed" + speedThreshold + "_matches" + patternFilePath._2 + ".csv"

  val domainSpecificStream: String = "maritime"

  val foldsNo: Int = 4
}
