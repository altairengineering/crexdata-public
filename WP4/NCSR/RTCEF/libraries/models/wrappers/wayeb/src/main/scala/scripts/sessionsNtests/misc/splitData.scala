package scripts.sessionsNtests.misc

import stream.array.EventStream
import stream.source.EmitMode
import stream.{GenericEvent, StreamFactory}
import scala.collection.mutable.ArrayBuffer

object splitData {
  def main(args: Array[String]): Unit = {
    val sampleInterval = 60
    val dataDir = "/mnt/Warehouse/data/maritime/imis/synopses/ais_jan2016_synopses_v0.8/csv/trajectories/" + sampleInterval + "sec"
    val filename = dataDir + "/" + sampleInterval + "sec_all.csv"
    val streamSource = StreamFactory.getDomainStreamSource(filename, "datacronMaritime", List.empty)
    val stream = streamSource.emitEventsAndClose(EmitMode.BUFFER).getStream
    val folds = 10

    val testSize = stream.size / folds

    val foldStreams = (1 to folds).map(f => (f, split(stream, testSize, f)))
    foldStreams.foreach(s => {
      s._2._1.writeCSV(dataDir + "/train_" + s._1 + ".csv", List("mmsi", "speed", "heading", "lon", "lat", "Timestamp"))
      s._2._2.writeCSV(dataDir + "/test_" + s._1 + ".csv", List("mmsi", "speed", "heading", "lon", "lat", "Timestamp"))
    })
  }

  def split(initialStreamBuffer: ArrayBuffer[GenericEvent], testSize: Int, offset: Int): (EventStream, EventStream) = {
    //val lefti = 0
    val leftj = (offset - 1) * testSize
    //val righti = leftj + 1
    //val rightj = testSize
    val testBuffer = initialStreamBuffer.slice(leftj, leftj + testSize)
    val trainBuffer = initialStreamBuffer.take(leftj) ++ initialStreamBuffer.takeRight(initialStreamBuffer.size - (leftj + testSize))
    val testStream = EventStream(testBuffer)
    val trainStream = EventStream(trainBuffer)
    (trainStream, testStream)
  }

}
