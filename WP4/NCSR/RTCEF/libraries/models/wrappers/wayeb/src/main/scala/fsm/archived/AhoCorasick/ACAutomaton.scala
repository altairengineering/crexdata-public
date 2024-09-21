package fsm.archived.AhoCorasick

import scala.collection.SortedMap
import scala.collection.mutable.Map
import scala.collection.mutable.Queue
import scala.collection.mutable.Set
import fsm.CountPolicy._

class ACAutomaton private[AhoCorasick] (keywords: List[String], extraSymbols: Set[Char]) {
  private val states = Map[Int, State]()
  private val inputSymbols = Set[Char]()
  private var countPolicy = OVERLAP

  /*def setCountPolicy(p: String): Int = {
    if (p.equalsIgnoreCase("overlap")) countPolicy = 0
    else if (p.equalsIgnoreCase("non-overlap")) countPolicy = 1
    else throw new IllegalArgumentException("Count policy for AC automaton cannot be " + p)
    countPolicy
  }*/

  private[AhoCorasick] def setCountPolicy(p: CountPolicy): CountPolicy = {
    if (p == OVERLAP | p == NONOVERLAP) countPolicy = p
    else throw new IllegalArgumentException("Count policy for AC automaton cannot be " + p)
    countPolicy
  }

  def build(): Unit = {
    buildSymbols()
    buildGoto()
    buildFail()
    buildDFA()
    checkCountPolicy()
  }

  def goto(state: Int, c: Char): Int = {
    if (!states.contains(state)) -1
    else states(state).goto(c)
  }

  private def buildSymbols(): Unit = {
    for (w <- keywords) {
      for (c <- w) {
        inputSymbols += c
      }
    }
    for (c <- extraSymbols) {
      inputSymbols += c
    }
  }

  private def buildGoto(): Unit = {
    var newState = 0
    states += (newState -> new State(newState))
    for (keyword <- keywords) {
      newState = enter(keyword, newState)
    }
    initLoop()
  }

  private def enter(keyword: String, oldState: Int): Int = {
    var state = 0
    var j = 0
    var c = keyword(j)
    while (goto(state, c) != -1) {
      state = goto(state, c)
      j += 1
      c = keyword(j)
    }
    var newState = oldState
    for (p <- j to (keyword.size - 1)) {
      newState += 1
      val newStateInst = new State(newState)
      states += (newState -> newStateInst)
      states(state).addTransition(keyword(p), newState)
      state = newState
    }
    states(state).addOutput(keyword)
    newState
  }

  private def initLoop(): Unit = {
    for (a <- inputSymbols) {
      if (goto(0, a) == -1) {
        states(0).addTransition(a, 0)
      }
    }
  }

  private def buildFail(): Unit = {
    val q = new Queue[Int]()
    val gtm0 = states(0).getGotoMap()
    for ((k, v) <- gtm0) {
      if (v != 0) {
        q.enqueue(v)
        states(v).setFail(0)
      }
    }
    var r = 0
    while (q.nonEmpty) {
      r = q.dequeue()
      val gtmr = states(r).getGotoMap()
      for ((a, s) <- gtmr) {
        if (states(r).goto(a) != -1) {
          q.enqueue(s)
          var state = states(r).getFail()
          var fs = states(state).goto(a)
          while (fs == -1) {
            state = states(state).getFail()
            fs = states(state).goto(a)
          }
          states(s).setFail(fs)
          states(s).addOutput(states(fs).getOutput())
        }

      }
    }
  }

  private def buildDFA(): Unit = {
    val q = new Queue[Int]()
    var s0 = 0
    for (a <- inputSymbols) {
      s0 = states(0).goto(a)
      states(0).addDelta(a, s0)
      if (s0 != 0) q.enqueue(s0)
    }
    var r = 0
    var sr = 0
    var fr = 0
    while (q.nonEmpty) {
      r = q.dequeue()
      for (a <- inputSymbols) {
        sr = states(r).goto(a)
        if (sr != -1) {
          q.enqueue(sr)
          states(r).addDelta(a, sr)
        } else {
          fr = states(r).getFail()
          states(r).addDelta(a, states(fr).getDelta(a))
        }
      }
    }
  }

  //TODO: if a keyword is a prefix of another keyword, then the latter will never be detected (raise warning)
  private def checkCountPolicy(): Unit = {
    if (countPolicy == NONOVERLAP) {
      val s0 = states(0)
      for ((k, sk) <- states) {
        if (sk.isFinal()) {
          sk.setDelta(s0.getDelta())
        }
      }
    }
  }

  def delta(s: Int, c: Char): Int = {
    var ns = 0
    ns = states(s).getDelta(c)
    if (ns == -1) ns = 0
    ns
  }

  def getOutput(s: Int): Set[String] = {
    states(s).getOutput()
  }

  def getStates: Map[Int, State] = states

  def getInputSymbols: Set[Char] = inputSymbols

  override def toString(): String = {
    var s = "Symbols: " + inputSymbols
    val sorted = SortedMap(states.toSeq: _*)
    for ((k, v) <- sorted) {
      s += "\n" + v.toString()
    }
    s
  }

  def dfaAsString(): String = {
    var v = Set[Int]()
    var d = ""
    var t = Set[Int]()
    val sorted = SortedMap(states.toSeq: _*)
    for ((key, value) <- sorted) {
      v += key
      d += "\nState: " + key + "\n" + value.deltaAsString()
      if (value.isFinal()) t += key
    }
    //println("V=" + v +"\nA=" + inputSymbols + "\nf=" + d + "\nT=" + t)
    "V=" + v + "\nA=" + inputSymbols + "\nf=" + d + "\nT=" + t
  }

}
