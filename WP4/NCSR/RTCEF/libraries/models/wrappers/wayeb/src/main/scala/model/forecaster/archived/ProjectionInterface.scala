package model.forecaster.archived

import fsm.SDFAInterface
import model.forecaster.ForecasterType.ForecasterType
import model.forecaster.runtime.RelativeForecast
import model.forecaster.{ForecasterInterface, ForecasterType}
import model.vmm.pst.{CyclicBuffer, PredictionSuffixTree}
import model.vmm.{Isomorphism, Symbol}
import model.waitingTime.ForecastMethod
import model.waitingTime.ForecastMethod.ForecastMethod
import stream.GenericEvent
import utils.SetUtils

class ProjectionInterface(
    fsm: SDFAInterface,
    pst: PredictionSuffixTree,
    iso: Isomorphism,
    val id: Int,
    method: ForecastMethod,
    threshold: Double,
    maxSpread: Int,
    horizon: Int,
    order: Int
) extends ForecasterInterface {

  private val buffer: CyclicBuffer = new CyclicBuffer(order + 1)
  private val permutations = SetUtils.permutationsAlt[Symbol](iso.getSymbols.toSet, horizon)(horizon)

  override def getNewForecast(
      state: Int,
      timestamp: Long
  ): RelativeForecast = {
    val storedSequence = buffer.pop
    val seqWithProbs = permutations.map(p => (p, estimateProbForFutureSequence(storedSequence, p)))
    val sortedByProb = seqWithProbs.toList.sortWith((x, y) => x._2 > y._2)
    val mostProbableFutureSequence = sortedByProb.head

    val futureSentences = mostProbableFutureSequence._1.reverse.map(s => iso.getMinTermForSymbol(s))
    val projectedStates = fsm.getNextStateWithSentences(state, futureSentences)

    val visitToFinal = findFirstVisitToFinal(projectedStates, 1)

    /*RelativePrediction( startRelativeToNow = pred.start,
      endRelativeToNow = pred.end,
      middle = pred.middle,
      prob = pred.prob,
      startRelativeToCounter = timestamp + pred.start,
      endRelativeToCounter = timestamp + pred.end,
      isPositive = pred.positive)*/

    RelativeForecast()
  }

  def addSymbol(event: GenericEvent): Unit = {
    val symbol = iso.evaluate(event)
    buffer.pushSymbol(symbol)
  }

  @scala.annotation.tailrec
  private def findFirstVisitToFinal(
      states: List[Int],
      i: Int
  ): Int = {
    states match {
      case Nil => -1
      case head :: tail => {
        if (fsm.isFinal(head)) i else findFirstVisitToFinal(tail, i + 1)
      }
    }
  }

  private def estimateProbForFutureSequence(
      storedSequence: List[Symbol],
      futureSequence: List[Symbol]
  ): Double = {
    pst.getConditionalProbFor(futureSequence, storedSequence)
  }

  override def getId: Int = id

  override def getMaxSpread: Int = maxSpread

  override def getStates: Set[Int] = fsm.getStates

  override def getType: ForecasterType = {
    if (ForecastMethod.isClassification(method)) ForecasterType.CLASSIFICATION
    else ForecasterType.REGRESSION
  }

}
