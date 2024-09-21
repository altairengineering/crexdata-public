package scripts.data.bio

import java.io.File
import stream.{GenericEvent, ResetEvent}
import com.github.tototoshi.csv.{CSVReader, CSVWriter}

object ProcessSims {
  def main(args: Array[String]): Unit = {

    val id = new File(Constants.pathToInterestingDir)
    val ifiles = id.listFiles().filter(_.isFile).toList.filter(f => f.getName.endsWith(".csv"))
    val nd = new File(Constants.pathToNotInterestingDir)
    val nfiles = nd.listFiles().filter(_.isFile).toList.filter(f => f.getName.endsWith(".csv"))

    val isims = ifiles.map(f => processSim(f.getAbsolutePath, f.getName, true))
    val nsims = nfiles.map(f => processSim(f.getAbsolutePath, f.getName, false))

    val itrainKeep = (Constants.percentTrain * isims.length).toInt
    val iTestKeep = isims.length - itrainKeep
    val ntrainKeep = (Constants.percentTrain * nsims.length).toInt
    val nTestKeep = nsims.length - ntrainKeep

    (0 until Constants.foldsNo).foreach(fold => {
      val ifrom = fold * iTestKeep
      val ito = (fold + 1) * iTestKeep
      println(ifrom + "-" + ito)
      val isimsTest = isims.slice(ifrom, ito)
      val isimsTrain = isims.take(ifrom) ::: isims.takeRight(isims.length - ito)
      println("\t" + isims.length + "/" + isimsTest.length + "/" + isimsTrain.length + "\n")

      val nfrom = fold * nTestKeep
      val nto = (fold + 1) * nTestKeep
      val nsimsTest = nsims.slice(nfrom, nto)
      val nsimsTrain = nsims.take(nfrom) ::: nsims.takeRight(nsims.length - nto)

      val trainSims = isimsTrain ::: nsimsTrain
      val testSims = isimsTest ::: nsimsTest
      writeSims(trainSims, Constants.dataDir + "/train" + (fold + 1) + ".csv")
      writeSims(testSims, Constants.dataDir + "/test" + (fold + 1) + ".csv")
    })

    /*val isimsTrain = isims.take(itrainKeep)
    val isimsTest = isims.drop(itrainKeep)


    val nsimsTrain = nsims.take(ntrainKeep)
    val nsimsTest = nsims.drop(ntrainKeep)

    val trainSims = isimsTrain ::: nsimsTrain
    val testSims = isimsTest ::: nsimsTest

    writeSims(trainSims,Constants.pathToTrainFile)
    writeSims(testSims,Constants.pathToTestFile)*/

  }

  private def writeSims(
      sims: List[List[GenericEvent]],
      fn: String
  ): Unit = {
    val writer = CSVWriter.open(fn)
    val simsit = sims.iterator
    var counter = 0
    while (simsit.hasNext) {
      val sim = simsit.next()
      writer.writeRow(List("-1", "r"))
      val eventit = sim.iterator
      while (eventit.hasNext) {
        counter += 1
        val event = eventit.next()
        val row = List(
          counter.toString,
          event.getValueOf("aliveSymbol"),
          event.getValueOf("necroticSymbol"),
          event.getValueOf("apoptoticSymbol"),
          event.getValueOf("aliveNo"),
          event.getValueOf("necroticNo"),
          event.getValueOf("apoptoticNo"),
          event.getValueOf("prevAliveNo"),
          event.getValueOf("prevNecroticNo"),
          event.getValueOf("prevApoptoticNo"),
          event.getValueOf("interesting"),
          (event.getValueOf("nextCETimestamp").toString.toInt + counter).toString,
          event.getValueOf("simId")
        )
        writer.writeRow(row)
      }
    }
    writer.close()
  }

  private def processSim(
      fn: String,
      simId: String,
      interesting: Boolean
  ): List[GenericEvent] = {
    val reader = CSVReader.open(fn)
    val it = reader.iterator
    it.next()
    var totalCounter = 0
    var prevEvent: GenericEvent = GenericEvent(0, "simpoint", 0, Map("aliveNo" -> 0, "necroticNo" -> 0, "apoptoticNo" -> 0, "nextCETimestamp" -> 1000000, "simId" -> "0"))
    var eventsList: List[GenericEvent] = List.empty
    while (it.hasNext) {
      val line = it.next()
      totalCounter += 1
      val flag =
        if (it.hasNext) 0
        else if (interesting) 1
        else 2
      val insertNextCE = true //if (interesting) false  else true
      val event = line2Event(line, simId, totalCounter, prevEvent, flag, insertNextCE)
      eventsList = event :: eventsList
      prevEvent = event
    }
    reader.close()
    eventsList.reverse
  }

  private def line2Event(
      line: Seq[String],
      simId: String,
      id: Int,
      previousEvent: GenericEvent,
      flag: Int,
      insertNextCE: Boolean
  ): GenericEvent = {
    val timestamp = line(0).toLong
    val eventType = "simpoint"
    val aliveSymbol = line(1)
    val necroticSymbol = line(2)
    val apoptoticSymbol = line(3)
    val aliveNo = line(4).toDouble
    val necroticNo = line(5).toDouble
    val apoptoticNo = line(6).toDouble
    val nextCETimestamp = if (insertNextCE) (49 - id) else 1000000
    val ge = GenericEvent(id, eventType, timestamp,
                          Map("aliveSymbol" -> aliveSymbol,
        "necroticSymbol" -> necroticSymbol,
        "apoptoticSymbol" -> apoptoticSymbol,
        "aliveNo" -> aliveNo,
        "necroticNo" -> necroticNo,
        "apoptoticNo" -> apoptoticNo,
        "prevAliveNo" -> previousEvent.getValueOf("aliveNo"),
        "prevNecroticNo" -> previousEvent.getValueOf("necroticNo"),
        "prevApoptoticNo" -> previousEvent.getValueOf("apoptoticNo"),
        "interesting" -> flag,
        "nextCETimestamp" -> nextCETimestamp,
        "simId" -> simId
      ))
    ge
  }

}
