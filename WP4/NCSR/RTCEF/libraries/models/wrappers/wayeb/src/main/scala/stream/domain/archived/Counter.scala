package stream.domain.archived

import scala.collection.immutable.Queue
import scala.collection.mutable.{Map, Set}

class Counter(
    pan: String,
    order: Int,
    lbls: Set[List[String]],
    symbols: Set[String]
) {
  private var counters = Map.empty[List[String], Map[String, Int]]
  private var labels = Set.empty[List[String]]
  private var suffix = Queue.empty[String]
  private var totalCounter = 0

  initCounters()

  private def initCounters(): Unit = {
    if (order == 0) {
      labels = Set(List.empty[String])
    } else {
      labels = lbls
    }
    for (label <- labels) {
      val thisCounters = Map.empty[String, Int]
      for (symbol <- symbols) {
        thisCounters += (symbol -> 0)
      }
      counters += (label -> thisCounters)
    }
  }

  def updateCounters(c: String): Unit = {
    if (suffix.size == order) {
      val label = getSuffixAsString
      val thisCounters = counters(label)
      thisCounters(c) += 1
      if (order != 0) {
        val (del, dq) = suffix.dequeue
        suffix = dq.enqueue(c)
      }
      totalCounter += 1
    } else {
      suffix = suffix.enqueue(c)
    }
  }

  private def getSuffixAsString: List[String] = {
    var sufstr = List.empty[String]
    if (order == 0) sufstr = List.empty[String]
    else {
      for (c <- suffix) {
        sufstr = sufstr ::: List(c)
      }
    }
    sufstr
  }

  def absorb(otherCounter: Counter): Unit = {
    for ((k1, v1) <- counters) {
      for ((k2, v2) <- v1) {
        val nc = otherCounter.getCounter(k1, k2)
        v1(k2) += nc
      }
    }
  }

  def getCounter(
      label: List[String],
      c: String
  ): Int = counters(label)(c)

  def getCounters: Map[List[String], Map[String, Int]] = counters

  override def toString: String = counters.toString()

  def getTotalCounter: Int = totalCounter
}
