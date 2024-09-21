package scripts.data.caviar

object ConfigFrameProcessing {
  val wayebHome: String = System.getenv("WAYEB_HOME")
  val dataDir: String = System.getenv("CAVIARDIR")
  val videoFilesDir: String = dataDir + "/videos"
  val sampling: Int = 5
  val patternFilePath: (String, String, String, Int, Int) =
    //(wayebHome + "/patterns/caviar/meeting/patternTest.sre", "overlap", "meetingInterval", 1, 4)
    //(wayebHome + "/patterns/caviar/meeting/initiation.sre", "nonoverlap", "meetingInitiation",10,4)
    //(wayebHome + "/patterns/caviar/meeting/context.sre", "nonoverlap", "meetingContext",1,4)
    (wayebHome + "/patterns/caviar/meeting/situation.sre", "nonoverlap", "meetingSituation", 100, 4)
  val pairedFile: String = dataDir + "/paired/pairedSampling" + sampling + "X" + patternFilePath._4 + ".csv"

}
