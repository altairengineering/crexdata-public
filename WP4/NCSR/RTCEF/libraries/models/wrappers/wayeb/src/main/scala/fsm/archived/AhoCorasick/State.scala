package fsm.archived.AhoCorasick

import scala.collection.SortedMap
import scala.collection.mutable.Map
import scala.collection.mutable.Set

class State(id: Int) {
  private val gotomap = Map[Char, Int]()
  private val delta = Map[Char, Int]()
  private var output = Set[String]()
  private var fail: Int = -1

  def addTransition(c: Char, nextState: Int): Unit = {
    if (!gotomap.contains(c))
      gotomap += (c -> nextState)
  }

  def addOutput(out: String): Unit = {
    output += out
  }

  def addOutput(out: Set[String]): Unit = {
    output = output ++ out
  }

  def addDelta(c: Char, nextState: Int): Unit = {
    delta += (c -> nextState)
  }

  def setDelta(d: Map[Char, Int]): Unit = {
    delta.clear()
    for ((k, v) <- d) {
      addDelta(k, v)
    }
  }

  def getDelta(c: Char): Int = {
    if (!delta.contains(c)) -1
    else delta(c)
  }

  def getDelta(): Map[Char, Int] = { delta }

  def getOutput(): Set[String] = { output }

  def goto(c: Char): Int = {
    if (!gotomap.contains(c)) -1
    else gotomap(c)
  }

  def getNextStates(except: List[Int]): List[Int] = {
    var lc = List[Int]()
    for ((k, v) <- gotomap) {
      if (!except.contains(v)) {
        lc = v :: lc
      }
    }
    lc
  }

  def getGotoMap(): Map[Char, Int] = { gotomap }

  def setFail(f: Int): Unit = {
    if (f >= 0) fail = f
  }

  def getFail(): Int = { fail }

  def isFinal(): Boolean = {
    if (output.nonEmpty) true
    else false
  }

  def deltaAsString(): String = {
    var s = ""
    val sorted = SortedMap(delta.toSeq: _*)
    for ((k, v) <- sorted) {
      s += "\n\t" + k + "\t|\t" + v
    }
    s
  }

  override def toString(): String = {
    var s = ""
    s += "State: " + id
    val sorted = SortedMap(gotomap.toSeq: _*)
    for ((k, v) <- sorted) {
      s += "\n\t" + k + "\t|\t" + v
    }
    s += "\n\tOutput: " + output
    s += "\n\tFail: " + fail
    s
  }

}
