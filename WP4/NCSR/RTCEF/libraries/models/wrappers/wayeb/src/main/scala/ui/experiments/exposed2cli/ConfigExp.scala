package ui.experiments.exposed2cli

import java.nio.file.{Files, Paths}

import fsm.CountPolicy
import model.waitingTime.ForecastMethod.ForecastMethod
import CountPolicy.CountPolicy
import model.waitingTime.ForecastMethod
import ui.ConfigUtils

import scala.io.Source

class ConfigExp(
                 val domain: String,
                 val patternFilePath: String,
                 val patternName: String,
                 val declarationsFilePath: String,
                 val resultsDir: String,
                 val orders: List[Int] = List(0),
                 val horizon: Int = 200,
                 val distances: List[(Double, Double)] = List((-1.0, -1.0)),
                 val maxSpreads: List[Int] = List(0),
                 val thresholds: List[Double] = List(0.5),
                 val foldsDir: String,
                 val folds: List[Int],
                 val spreadMethod: ForecastMethod = ForecastMethod.SMARTSCAN,
                 val finalsEnabled: Boolean = false,
                 val policy: CountPolicy = CountPolicy.NONOVERLAP,
                 val maxNoStatesList: List[Int] = List(1000),
                 val maxSize: Int = 2000,
                 val pMins: List[Double] = List(0.0),
                 val alphas: List[Double] = List(0.0),
                 val gammas: List[Double] = List(0.0),
                 val rs: List[Double] = List(0.0),
                 val target: String = "ce",
                 val wt: String = "normal",
                 val wtCutoffThreshold: Double = ConfigUtils.wtCutoffThreshold
               ) {

  override def toString: String = {

    var pattern = ""
    val patPath = Paths.get(patternFilePath)
    if (Files.exists(patPath) & !patternFilePath.equalsIgnoreCase("")) {
      val bufferedSourcePattern = Source.fromFile(patternFilePath)
      pattern = bufferedSourcePattern.getLines().mkString
      bufferedSourcePattern.close()
    }

    var decl = ""
    val declPath = Paths.get(declarationsFilePath)
    if (Files.exists(declPath) & !declarationsFilePath.equalsIgnoreCase("")) {
      val bufferedSourceDecl = Source.fromFile(declarationsFilePath)
      decl = bufferedSourceDecl.getLines().mkString
      bufferedSourceDecl.close()
    }

    val str =
      "domain: " + domain + "\n" +
        "patternFilePath: " + patternFilePath + "\n" +
        "patternName: " + patternName + "\n" +
        "pattern: " + pattern +
        "declarationsFilePath" + declarationsFilePath + "\n" +
        "declarations: " + decl + "\n" +
        "resultsDir: " + resultsDir + "\n" +
        "orders: " + orders + "\n" +
        "horizon: " + horizon + "\n" +
        "distances: " + distances + "\n" +
        "maxSpreads: " + maxSpreads + "\n" +
        "thresholds: " + thresholds + "\n" +
        "foldsDir: " + foldsDir + "\n" +
        "folds: " + folds + "\n" +
        "spreadMethod: " + spreadMethod + "\n" +
        "finalsEnabled: " + finalsEnabled + "\n" +
        "policy: " + policy + "\n" +
        "maxNoStatesList: " + maxNoStatesList + "\n" +
        "maxSize: " + maxSize + "\n" +
        "pMins: " + pMins + "\n" +
        "alphas: " + alphas + "\n" +
        "gammas: " + gammas + "\n" +
        "rs: " + rs + "\n" +
        "target: " + target + "\n" +
        "wt: " + wt + "\n" +
        "wtCutoffThreshold: " + wtCutoffThreshold + "\n"

    str
  }

}
