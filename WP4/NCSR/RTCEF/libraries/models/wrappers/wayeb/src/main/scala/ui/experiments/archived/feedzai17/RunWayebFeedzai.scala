package ui.experiments.archived.feedzai17

import java.io.File
import java.nio.file.{Files, Paths}

import engine.ERFEngine
import stream.domain.archived.FeedzaiStream
import ui.ConfigUtils
import workflow.provider.source.dfa.DFASourceFromXML
import workflow.provider.source.matrix.MCSourceProbs
import workflow.provider.source.forecaster.ForecasterSourceBuild
import workflow.provider.source.wt.WtSourceMatrix
import workflow.provider._

import scala.xml.XML

object RunWayebFeedzai {
  def main(args: Array[String]): Unit = {
    //val home = System.getenv("WAYEB_HOME")

    val inputFileName = args(0)
    val trxsNo = args(1).toInt
    val learnPrecentage = args(2).toDouble
    val configFileName = args(3)
    val fz_cnf = new FeedzaiConfigurator(configFileName)
    //println(fz_cnf.toString())
    val horizon = fz_cnf.getHorizon()
    val predThresholds = fz_cnf.getPredictionThresholds()
    val maxSpreads = fz_cnf.getMaxSpreads()
    val order = fz_cnf.getOrder()
    val patterns = fz_cnf.getPatterns()
    val resultsDir = fz_cnf.getResultsDir()
    val expirationTime = fz_cnf.getExpirationTime()

    val file = new File(resultsDir)
    deleteRecursively(file)
    val rdir = new File(resultsDir)
    rdir.mkdir()

    val t1 = System.nanoTime()

    for (p <- patterns) {
      val name = p._1
      val seq = p._2
      val policy = p._3
      val partitionAttribute = "pan"
      println("\n\n\t\tTime to predict fraudulent transactions for the pattern: " + name + "/" + seq + "/" + policy + "\n\n")
      val fzstream = new FeedzaiStream(inputFileName, trxsNo, learnPrecentage, name)
      fzstream.setConfig(fz_cnf.getFarAwayWindow(), fz_cnf.getBigThreshold(), fz_cnf.getSmallThreshold(), fz_cnf.getBigAfterSmallWindow(), fz_cnf.getFlashAttackWindow())
      fzstream.generateStream()
      fzstream.calculateProbs(order)
      fzstream.writeModel(resultsDir + "/model" + name + ".wb")
      val tprobs = fzstream.getProbs
      val testStream = fzstream.getTestStream
      //val dfa = RegExpUtils.xml2dfa(seq,policy,order,testStream.getEventTypes())
      val dfaProvider = DFAProvider(new DFASourceFromXML(seq, policy, order, testStream.getEventTypes))
      val fsmp = FSMProvider(dfaProvider)
      val mcp = MarkovChainProvider(new MCSourceProbs(fsmp, tprobs))
      for (predThres <- predThresholds) {
        for (maxSpread <- maxSpreads) {
          val wtdProv = WtProvider(WtSourceMatrix(fsmp, mcp, horizon, ConfigUtils.defaultFinalsEnabled))
          val pp = ForecasterProvider(ForecasterSourceBuild(fsmp                = fsmp,
                                                          wtdp                = wtdProv,
                                                          horizon             = horizon,
                                                          confidenceThreshold = predThres,
                                                          maxSpread           = maxSpread
          ))
          val ace = ERFEngine(fsmp, pp, true, -1, true, false, (-1.0, -1.0), true)
          val profiler = ace.processStream(testStream)
          var seqStr = seq
          if (Files.exists(Paths.get(seq))) {
            val loadnode = XML.loadFile(seq)
            seqStr = (loadnode \\ "name").text
          }
          val resultsFile = resultsDir + "/" + name + "_seq_" + seqStr + "_pol_" + policy + "_ord_" + order
          profiler.printProfileInfo(resultsFile)
        }
      }
    }

    val t2 = System.nanoTime()
    val elapsed = (t2 - t1) / 1000000000
    println("Finished experiments in " + elapsed + " seconds.")

  }

  def deleteRecursively(file: File): Unit = {
    if (file.isDirectory)
      file.listFiles.foreach(deleteRecursively)
    if (file.exists && !file.delete)
      throw new Exception(s"Unable to delete ${file.getAbsolutePath}")
  }
}
