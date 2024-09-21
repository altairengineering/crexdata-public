package ui.experiments.archived.feedzai17

import fsm.CountPolicy._

import scala.collection.mutable.Set
import scala.xml._

class FeedzaiConfigurator(filename: String) {
  val loadnode = XML.loadFile(filename)

  val horizon = (loadnode \\ "horizon").text.toInt
  val order = (loadnode \\ "order").text.toInt
  val expirationTime = (loadnode \\ "expirationTime").text.toInt
  val predThresholdNodes = (loadnode \\ "predictionThreshold")
  val predictionThresholds = Set[Double]()
  for (tnode <- predThresholdNodes) {
    predictionThresholds += tnode.text.toDouble
  }
  val maxSpreadNodes = (loadnode \\ "maxSpread")
  val maxSpreads = Set[Int]()
  for (mnode <- maxSpreadNodes) {
    maxSpreads += mnode.text.toInt
  }
  val resultsDir = (loadnode \\ "resultsDir").text

  val patternNodes = (loadnode \\ "fsm/classical/pattern")
  val patterns = Set[Tuple4[String, String, CountPolicy, String]]()
  for (pnode <- patternNodes) {
    val name = (pnode \\ "name").text
    val seq = (pnode \\ "sequence").text
    val policyStr = (pnode \\ "policy").text
    var policy = NONOVERLAP
    if (policyStr.equalsIgnoreCase("overlap")) policy = OVERLAP
    val partitionAttribute = "pan"
    val patternTuple: Tuple4[String, String, CountPolicy, String] = (name.toString, seq.toString, policy, partitionAttribute)
    patterns += patternTuple
  }

  val farAwayWindow = (loadnode \\ "farAwayWindow").text.toLong
  val bigAfterSmallWindow = (loadnode \\ "bigAfterSmallWindow").text.toLong
  val flashAttackWindow = (loadnode \\ "flashAttackWindow").text.toLong
  val bigThreshold = (loadnode \\ "bigThreshold").text.toDouble
  val smallThreshold = (loadnode \\ "smallThreshold").text.toDouble

  /*val chars = (loadnode \\ "char")
  val eventCharsAllowed = Set[Char]()
  for (c <- chars) {
    eventCharsAllowed += c.text.toString.head
  }*/

  def getHorizon(): Int = { horizon }
  def getOrder(): Int = { order }
  def getExpirationTime(): Int = { expirationTime }
  def getPredictionThresholds(): Set[Double] = { predictionThresholds }
  def getMaxSpreads(): Set[Int] = { maxSpreads }
  def getResultsDir(): String = { resultsDir }
  def getPatterns(): Set[Tuple4[String, String, CountPolicy, String]] = { patterns }
  def getFarAwayWindow(): Long = { farAwayWindow }
  def getBigAfterSmallWindow(): Long = { bigAfterSmallWindow }
  def getFlashAttackWindow(): Long = { flashAttackWindow }
  def getBigThreshold(): Double = { bigThreshold }
  def getSmallThreshold(): Double = { smallThreshold }
  //def getEventCharsAllowed(): Set[Char] = {eventCharsAllowed}

  override def toString(): String = {
    var s = horizon + "\n" + order + "\n" + expirationTime + "\n" + predictionThresholds + "\n" + maxSpreads
    s += "\n" + resultsDir
    s += "\n" + patterns
    s
  }

}
