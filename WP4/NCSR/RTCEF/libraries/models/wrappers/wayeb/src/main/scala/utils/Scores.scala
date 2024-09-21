package utils

import breeze.numerics.sqrt

import scala.math.log10

object Scores {
  /**
    * Estimates logarithm of a number with a given base.
    *
    * @param x The number.
    * @param base The base.
    * @return The logarithm.
    */
  def getMetrics(
               tp: Int,
               tn: Int,
               fp: Int,
               fn: Int
             ): Map[String, Double] = {
    val tpfp = tp + fp
    val precision = if (tpfp != 0) tp.toDouble / tpfp else -1
    val tpfn = tp + fn
    val recall = if (tpfn != 0) tp.toDouble / tpfn else -1
    val f1 = if ((precision != -1 & recall != -1) & precision+recall != 0.0 ) (2 * precision * recall) / (precision + recall) else -1
    val tnfp = tn + fp
    val specificity = if (tnfp != 0) tn.toDouble / tnfp else -1
    val total = tp + tn + fp + fn
    val accuracy = if (total != 0) (tp + tn).toDouble / total else -1
    val tnfn = tn + fn
    val npv = if (tnfn != 0) tn.toDouble / tnfn else -1
    // do not use this formula to calculate mcc, denominator can easily overflow if the number of predictions is large
    //val mcc = if (tpfp != 0 & tpfn != 0 & tnfp != 0 & tnfn != 0) ((tp * tn).toDouble - (fp * fn) )/( math.sqrt(tpfp * tpfn * tnfp * tnfn)) else -1
    val mcc =
      if (tpfp == 0 | tpfn == 0 | tnfp == 0 | tnfn == 0) 0.0
      else {
        val fdr = 1 - precision
        val fnr = 1 - recall
        val fpr = 1 - specificity
        val fomr = 1 - npv
        math.sqrt(precision * recall * specificity * npv) - math.sqrt(fdr * fnr * fpr * fomr)
      }

    if (tpfp == 0 | tpfn == 0 | tnfp == 0 | tnfn == 0) 0.0
    else {
      val fdr = 1 - precision
      val fnr = 1 - recall
      val fpr = 1 - specificity
      val fomr = 1 - npv
      math.sqrt(precision * recall * specificity * npv) - math.sqrt(fdr * fnr * fpr * fomr)
    }

    Map[String, Double](
      "mcc" -> mcc,
      "precision" -> precision,
      "recall" -> recall,
      "f1" -> f1
    )
  }
}
