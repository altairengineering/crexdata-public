package scripts.data.caviar

import com.github.tototoshi.csv.CSVWriter
import scala.math.{sqrt, acos, toDegrees}

object PairMatching {

  def CreatePairedStream(
      videoFileNames: List[String],
      outFileName: String,
      sampling: Int
  ): Unit = {
    var pairs: Set[(Int, Int)] = Set.empty
    val writer = CSVWriter.open(outFileName)
    var lastSeen: Map[Int, CaviarObject] = Map.empty
    var lastSeenPair: Map[(Int, Int), Int] = Map.empty
    var frameNo = 0
    for (videofn <- videoFileNames) {
      val bufferedSource = io.Source.fromFile(videofn)
      val lines = bufferedSource.getLines()
      for (line <- lines) {
        frameNo += 1
        val objectsAndGroups = line.split(":")
        val objectsString = objectsAndGroups(0)
        val objectsSplit = objectsString.split(";")
        var cos: Map[Int, CaviarObject] = Map.empty
        for (objectStr <- objectsSplit) {
          val attrs = objectStr.split("\\|")
          val frame = s"${attrs(1)}".toInt
          val id = s"${attrs(2)}".toInt
          val x = s"${attrs(3)}".toInt
          val y = s"${attrs(4)}".toInt
          val situation = s"${attrs(5)}".toString
          val context = s"${attrs(6)}".toString
          val gaze = s"${attrs(7)}".toInt
          val co =
            if (lastSeen.contains(id)) {
              val lastFrame = lastSeen(id).frame
              if (lastFrame == frameNo - 1) {
                val prevx = lastSeen(id).x
                val prevy = lastSeen(id).y
                val diffx = x - prevx
                val diffy = y - prevy
                //val refx = 1
                //val refy = 0
                val theta =
                  if (diffy >= 0) toDegrees(acos(diffx / magnitude(diffx, diffy)))
                  else 180 + toDegrees(acos(diffx / magnitude(diffx, diffy)))
                CaviarObject(id, frameNo, x, y, diffx, diffy, theta.toInt, gaze, situation, context)
              } else CaviarObject(id, frameNo, x, y, 0, 0, 0, gaze, situation, context)
            } else {
              CaviarObject(id, frameNo, x, y, 0, 0, 0, gaze, situation, context)
            }

          cos = cos + (id -> co)
          lastSeen += (id -> co)
        }
        var cgs: Map[(Int, Int), CaviarGroup] = Map.empty
        if (objectsAndGroups.length == 2) {
          val groupsString = objectsAndGroups(1)
          val groupsSplit = groupsString.split(";")
          for (groupStr <- groupsSplit) {
            val attrs = groupStr.split("\\|")
            val frame = s"${attrs(1)}".toInt
            val ids = s"${attrs(2)}".split(",").map(_.toInt).toSet
            val idPermutations = utils.SetUtils.permutations(ids, 2).map(x => (x.head, x.tail.head)).filter(p => p._1 != p._2)
            for (idperm <- idPermutations) {
              val id1 = idperm._1
              val id2 = idperm._2
              val situation = s"${attrs(3)}"
              val context = s"${attrs(4)}"
              val cg = CaviarGroup(id1, id2, frameNo, situation, context)
              cgs = cgs + ((id1, id2) -> cg)
            }
          }
        }

        val ids = cos.keySet
        val idPermutations = utils.SetUtils.permutations(ids, 2).map(x => (x.head, x.tail.head)).filter(p => p._1 != p._2)
        val existingPairs = idPermutations & pairs
        val newPermutations = idPermutations &~ (existingPairs | existingPairs.map(p => (p._2, p._1)))
        var newPairs: Set[(Int, Int)] = Set.empty
        var newPermutationsToCheck = newPermutations
        while (newPermutationsToCheck.nonEmpty) {
          val newPerm = newPermutationsToCheck.head
          val newReversedPerm = (newPerm._2, newPerm._1)
          newPairs = newPairs + newPerm
          newPermutationsToCheck = newPermutationsToCheck - newPerm
          newPermutationsToCheck = newPermutationsToCheck - newReversedPerm
        }
        pairs = pairs ++ newPairs
        val pairs2write = existingPairs ++ newPairs
        for (pair <- pairs2write) {
          val co1 = cos(pair._1)
          val co2 = cos(pair._2)
          if (shouldWrite(lastSeenPair, co1.frame, pair, sampling)) {
            val objectRow = List(co1.frame.toString, pair._1.toString, pair._2.toString, co1.toString, co2.toString)
            //val objectRow = List(co1.frame.toString, co1.toString, co2.toString)
            val x1 = co1.x
            val y1 = co1.y
            val x2 = co2.x
            val y2 = co2.y
            val distance = magnitude(x2 - x1, y2 - y1).toInt //sqrt( (x2 - x1) * (x2 - x1)  + (y2 - y1) * (y2 - y1) )
            val groupRow =
              if (cgs.contains(pair)) {
                val cg = cgs(pair)
                List(distance.toString + ";" + cg.situation + ";" + cg.context)
              } else if (cgs.contains((pair._2, pair._1))) {
                val cg = cgs((pair._2, pair._1))
                List(distance.toString + ";" + cg.situation + ";" + cg.context)
              } else List(distance.toString)
            val row = objectRow ::: groupRow
            if (lastSeenPair.contains(pair)) {
              val lastSeen = lastSeenPair(pair)
              if (lastSeen != frameNo - sampling) writer.writeRow(List("-1", pair._1, pair._2))
            } else writer.writeRow(List("-1", pair._1, pair._2))
            lastSeenPair += (pair -> frameNo)
            //println(row)
            writer.writeRow(row)
          }
        }
      }
      lastSeenPair = Map.empty
    }
    writer.close()

  }

  private def shouldWrite(
      lastSeenPair: Map[(Int, Int), Int],
      frame: Int,
      pair: (Int, Int),
      sampling: Int
  ): Boolean = {
    if (lastSeenPair.contains(pair)) {
      val lastSeenFrame = lastSeenPair(pair)
      if (frame - lastSeenFrame == sampling) true
      else false
    } else true
  }

  private def magnitude(x: Int, y: Int): Double = {
    sqrt(x * x + y * y.toDouble)
  }

}
