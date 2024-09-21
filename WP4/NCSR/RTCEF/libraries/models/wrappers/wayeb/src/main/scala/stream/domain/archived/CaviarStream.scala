package stream.domain.archived

import java.util

import stream.GenericEvent
import stream.array.{EventStream, EventStreamI}
//import streamer.FStreamer

/**
  * Created by manosnt on 7/8/19.
  */

object CaviarStream {
  def apply(filename: String): CaviarStream = new CaviarStream(filename)
}

class CaviarStream(filename: String) extends EventStreamI {
  override def generateStream(): EventStream = {

    val eventStream = new EventStream()
    var totalCounter = 1
    val bufferedSource = io.Source.fromFile(filename)
    val lines = bufferedSource.getLines()
    //val eventsStart = parseHeader(lines.next())
    var pairsId = scala.collection.mutable.Map[String, Int]()
    var pairId = 0
    for (aline <- lines) {
      line2Events(aline)
    }

      def line2Events(line: String): Unit = {

        val objectsAndGroups = line.split(":")
        val objects = objectsAndGroups(0)
        //var ObjectEvents = new ListBuffer[ObjectEvent]()
        var ObjectEvents = new util.HashMap[Int, ObjectEvent]()

        if (objectsAndGroups.length == 2) {
          val groups = objectsAndGroups(1)

          val objectlist = objects.split(";")
          for (obj <- objectlist) {
            val attrs = obj.split("\\|")
            //val frame = s"${attrs(1)}".toInt
            val id = s"${attrs(2)}".toInt
            val x = s"${attrs(3)}".toInt
            val y = s"${attrs(4)}".toInt
            val cxt = s"${attrs(6)}".toString
            val gaze = s"${attrs(7)}".toInt

            ObjectEvents.put(id, new ObjectEvent(gaze, x, y, id, cxt))
            //ObjectEvents += new ObjectEvent(frame, gaze, x, y, id)

          }

          val grouplist = groups.split(";")
          for (gr <- grouplist) {
            val attrs = gr.split("\\|")
            val frame = s"${attrs(1)}".toInt
            val ids = s"${attrs(2)}".split(",").map(_.toInt)
            val situation = s"${attrs(3)}"
            val cxt = s"${attrs(4)}"

            for (i <- 0 until ids.size - 1)
              for (j <- i + 1 until ids.size)
                if (ObjectEvents.containsKey(ids(i)) && ObjectEvents.containsKey(ids(j))) {
                  val obj1 = ObjectEvents.get(ids(i))
                  val obj2 = ObjectEvents.get(ids(j))
                  val ge = GenericEvent(totalCounter, "SampledCritical", frame,
                                        Map("id" -> { ids(i) + " " + ids(j) }, "gaze1" -> obj1.gaze, "gaze2" -> obj2.gaze, "x1" -> obj1.x, "y1" -> obj1.y, "x2" -> obj2.x, "y2" -> obj2.y, "situation" -> situation, "cxt" -> cxt,
                      "cxt1" -> obj1.cxt, "cxt2" -> obj2.cxt))

                  totalCounter += 1
                  eventStream.addEvent(ge)
                }
          }

        }

      }

    eventStream
  }

  class ObjectEvent(gz: Int, xc: Int, yc: Int, _id: Int, context: String) {
    val gaze = gz
    val x = xc
    val y = yc
    val id = _id
    val cxt = context

  }

}

