package ui.experiments.archived.text

import com.github.tototoshi.csv.{CSVReader, CSVWriter}
import com.typesafe.scalalogging.LazyLogging
import fsm.symbolic.sfa.logic.{AtomicSentence, PredicateConstructor, Sentence}
import stream.StreamFactory
import ui.experiments.exposed2cli.PatternExperimentsSPSA.logger
import model.vmm.pst.{CSTLearner, PSTLearner}
import model.vmm.{Isomorphism, Symbol}

object RunPSTTest extends LazyLogging {

  def main(args: Array[String]): Unit = {
    val alphabet: List[String] = List("A", "B", "C", "D", "R") ::: (1 to 251).toList.map(x => x.toString)
    val symbols = (1 to alphabet.length).map(i => Symbol(i)).toList
    val asymbol = symbols(0)
    val bsymbol = symbols(1)
    val csymbol = symbols(2)
    val dsymbol = symbols(3)
    val rsymbol = symbols(4)
    val simpleSentences = alphabet.map(s => AtomicSentence(PredicateConstructor.getEventTypePred(s)).asInstanceOf[Sentence])
    val sentences = simpleSentences
    val iso = Isomorphism(sentences, symbols)
    logger.info("ISO: " + iso.toString)

    val str = List("A", "B", "R", "A", "C", "A", "D", "A", "B", "R", "A")
    val trainStream = StreamFactory.getStreamSource(str)
    val maxOrder = 12
    val partitionAttribute = "$"

    logger.info("Learning counter suffix tree")
    val cstLearner = new CSTLearner(maxOrder, iso, partitionAttribute)
    trainStream.emitEventsToListener(cstLearner)
    val cst = cstLearner.getCST
    logger.info("Done with learning counter suffix tree")
    logger.info(cst.toString)
    logger.info("ISO: " + cst.getSymbols.map(s => (s, iso.getMinTermForSymbol(s))).toList)
    cst.print(iso)

    val cadabra = List(csymbol, asymbol, dsymbol, asymbol, bsymbol, rsymbol, asymbol).reverse
    val pMin = 0.001
    val alpha = 0.01
    val gamma = 0.001
    val r = 1.05
    val psTLearner = PSTLearner(symbols.toSet, maxOrder, pMin, alpha, gamma, r)

    logger.info("\tLEARNING WITH NEW METHOD\n")
    logger.info("Learning prediction suffix tree")
    val pst1 = psTLearner.learnVariant(cst, false)
    logger.info("Done with learning counter suffix tree")
    logger.info("PST size: " + pst1.getSize)
    //logger.info(pst1.toString)
    //pst1.print(iso,0.005)
    logger.info("logeval NEW: " + pst1.logEval(cadabra))
    logger.info("logloss NEW: " + pst1.avgLogLoss(cadabra))
    val pCgivenABRA1 = pst1.getConditionalProbFor(csymbol, List(asymbol, bsymbol, rsymbol, asymbol).reverse)
    logger.info("p(c|abra) NEW: " + pCgivenABRA1)

    logger.info("\tLEARNING WITH OLD METHOD\n")
    logger.info("Learning prediction suffix tree")
    val pst = psTLearner.learnOriginal(cst, true)
    logger.info("Done with learning counter suffix tree")
    logger.info("PST size: " + pst.getSize)
    //logger.info(pst.toString)
    //pst.print(iso,0.005)
    logger.info("logeval OLD: " + pst.logEval(cadabra))
    logger.info("logloss OLD: " + pst.avgLogLoss(cadabra))
    val pCgivenABRA = pst.getConditionalProbFor(csymbol, List(asymbol, bsymbol, rsymbol, asymbol).reverse)
    logger.info("p(c|abra) OLD: " + pCgivenABRA)

    /*val canodeDist = SymbolDistribution(Map[Symbol,Double](asymbol -> 0.001,bsymbol -> 0.001,csymbol -> 0.001,dsymbol -> 0.996,rsymbol -> 0.001))
    val canode = PredictionSuffixTree(List(asymbol,csymbol),canodeDist)

    val danodeDist = SymbolDistribution(Map[Symbol,Double](asymbol -> 0.001,bsymbol -> 0.996,csymbol -> 0.001,dsymbol -> 0.001,rsymbol -> 0.001))
    val danode = PredictionSuffixTree(List(asymbol,dsymbol),danodeDist)

    val ranodeDist = SymbolDistribution(Map[Symbol,Double](asymbol -> 0.001,bsymbol -> 0.001,csymbol -> 0.996,dsymbol -> 0.001,rsymbol -> 0.001))
    val ranode = PredictionSuffixTree(List(asymbol,rsymbol),ranodeDist)

    val anodeDist = SymbolDistribution(Map[Symbol,Double](asymbol -> 0.001,bsymbol -> 0.498,csymbol -> 0.25,dsymbol -> 0.25,rsymbol -> 0.001))
    val anode = PredictionSuffixTree(List(asymbol),anodeDist,Map[Symbol,PredictionSuffixTree](csymbol -> canode, dsymbol -> danode, rsymbol -> ranode))

    val bnodeDist = SymbolDistribution(Map[Symbol,Double](asymbol -> 0.001,bsymbol -> 0.001,csymbol -> 0.001,dsymbol -> 0.001,rsymbol -> 0.996))
    val bnode = PredictionSuffixTree(List(bsymbol),bnodeDist)

    val cnodeDist = SymbolDistribution(Map[Symbol,Double](asymbol -> 0.996,bsymbol -> 0.001,csymbol -> 0.001,dsymbol -> 0.001,rsymbol -> 0.001))
    val cnode = PredictionSuffixTree(List(csymbol),cnodeDist)

    val dnodeDist = SymbolDistribution(Map[Symbol,Double](asymbol -> 0.996,bsymbol -> 0.001,csymbol -> 0.001,dsymbol -> 0.001,rsymbol -> 0.001))
    val dnode = PredictionSuffixTree(List(dsymbol),dnodeDist)

    val rnodeDist = SymbolDistribution(Map[Symbol,Double](asymbol -> 0.996,bsymbol -> 0.001,csymbol -> 0.001,dsymbol -> 0.001,rsymbol -> 0.001))
    val rnode = PredictionSuffixTree(List(rsymbol),rnodeDist)

    val epsilonDist = SymbolDistribution(Map[Symbol,Double](asymbol -> 0.45,bsymbol -> 0.183,csymbol -> 0.092,dsymbol -> 0.092,rsymbol -> 0.183))
    val root = PredictionSuffixTree(List.empty,epsilonDist,Map[Symbol,PredictionSuffixTree](asymbol -> anode, bsymbol -> bnode, csymbol -> cnode, dsymbol -> dnode, rsymbol -> rnode))

    val cmp = root.compare(pst1,0.005)

    logger.info("Comparison: " + cmp)*/

    //val suffix = List(asymbol,rsymbol)
    //val suffixNode = pst1.getNodeFor(suffix)
    //logger.info(suffixNode.toString)

    val testfn = "/home/zmithereen/data/text/stream/folds/fold1_test.csv"
    val trainfn = "/home/zmithereen/data/text/stream/folds/fold1_train.csv"
    val resultsFile = "/home/zmithereen/data/text/results/hyperparameter.csv"
    /*val pMin = 0.001
    val alpha = 0.0
    val gamma = 0.0001
    val r = 1.01
    val maxOrder = 3//7*/
    //val pMinStep = 0.0005
    //val alphaStep = 0.005
    //val gammaStep = 0.00005
    //val rstep = 0.005
    //val pMins: List[Double] = (1 to 4).toList.map(x => x*pMinStep)
    //val alphas: List[Double] = List(0.0)
    //val gammas: List[Double] = (1 to 4).toList.map(x => x*gammaStep)
    //val rs: List[Double] = (1 to 4).toList.map(x => (x*rstep) + 1)
    val pMins = List(0.0001, 0.001, 0.01)
    val alphas = List(0.0)
    val gammas = List(0.0001, 0.001, 0.01)
    val rs = List(1.005, 1.01, 1.02, 1.03, 1.04, 1.05)
    val maxOrders: List[Int] = List(1, 2, 3, 4, 5, 6, 7)
    testOnText(trainfn, testfn, pMins, alphas, gammas, rs, maxOrders, resultsFile)
  }

  def testOnText(
      trainfn: String,
      testfn: String,
      pMins: List[Double],
      alphas: List[Double],
      gammas: List[Double],
      rs: List[Double],
      maxOrders: List[Int],
      resultsFile: String
  ): Unit = {
    logger.info("test on text")
    val partitionAttribute = "$"
    val alphabet: List[String] = List("#", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z") ::: (1 to 229).toList.map(x => x.toString)
    val symbols = alphabet.indices.map(i => Symbol(i)).toList
    val simpleSentences = alphabet.map(s => AtomicSentence(PredicateConstructor.getEventTypePred(s)).asInstanceOf[Sentence])
    val sentences = simpleSentences
    val iso = Isomorphism(sentences, symbols)

    val train = txt2list(trainfn).reverse
    val test = txt2list(testfn)
    //val train = txt.take((0.9 * txt.length).toInt)
    //val test = txt.takeRight((0.1 * txt.length).toInt)
    val testSymbols = test.map(x => symbols(alphabet.indexOf(x)))
    val trainStream = StreamFactory.getStreamSource(train)

    var bestLogLoss: Double = 10000.0
    var bestParValues: Map[String, String] = Map.empty
    val writer = CSVWriter.open(resultsFile)
    val header = List("logloss", "maxOrder", "pMin", "alpha", "gamma", "r", "size")
    writer.writeRow(header)
    for (
      pMin <- pMins;
      alpha <- alphas;
      gamma <- gammas;
      r <- rs;
      maxOrder <- maxOrders
    ) {
      logger.info("Learning counter suffix tree")
      logger.info("maxOrder/pMin/alpha/gamma/r\t" + maxOrder + "/" + pMin + "/" + alpha + "/" + gamma + "/" + r)
      val cstLearner = new CSTLearner(maxOrder, iso, partitionAttribute)
      trainStream.emitEventsToListener(cstLearner)
      val cst = cstLearner.getCST
      logger.info("Learning prediction suffix tree")
      val psTLearner = PSTLearner(symbols.toSet, maxOrder, pMin, alpha, gamma, r)

      val pst1 = psTLearner.learnVariant(cst, false)
      val logloss = pst1.avgLogLoss(testSymbols)
      val size = pst1.getSize
      logger.info("logeval: " + pst1.logEval(testSymbols))
      logger.info("logloss: " + pst1.avgLogLoss(testSymbols))
      logger.info("PST size: " + size)
      val row: List[String] = List(logloss.toString, maxOrder.toString, pMin.toString, alpha.toString, gamma.toString, r.toString, size.toString)
      if (logloss < bestLogLoss) {
        bestLogLoss = logloss
        bestParValues = header.zip(row).toMap
      }
      logger.info("\n\n BEST THUS FAR: " + bestParValues + "\n\n")
      writer.writeRow(row)
      //logger.info("OLD")
      //val pst = psTLearner.learn(cst)
      //logger.info("logeval: " + pst.logEval(testSymbols))
      //logger.info("logloss: " + pst.avgLogLoss(testSymbols))
      //logger.info("PST size: " + pst.getSize)
    }
    logger.info("best: " + bestParValues)
    writer.close()

  }

  def txt2list(fn: String): List[String] = {
    val reader = CSVReader.open(fn)
    val it = reader.iterator
    var result: List[String] = List.empty
    while (it.hasNext) {
      val line = it.next()
      result = line(1) :: result
    }
    reader.close()
    result
  }

}
