package Checks

import org.scalacheck._
import Gen._
import fsm.classical.pattern.regexp.OperatorType._
import fsm.classical.pattern.regexp.NodeType._
import fsm.classical.pattern.regexp.{OperatorNode, RegExpTree, RegExpUtils, SymbolNode}
import stream.source.EmitMode
import stream.StreamFactory
import ui.ConfigUtils
import workflow.provider.source.dfa.DFASourceRegExp
import workflow.provider.{DFAProvider, FSMProvider}
import workflow.task.engineTask.ERFTask

import scala.collection.mutable.Map

object DisSpec extends Properties("Disambiguator") {
  import Prop.forAll

  val symbols = Set('a', 'b', 'c')
  //implicit val arbChar = Arbitrary.arbitrary[Char] suchThat(symbols.contains(_))

  class MyChar(val c: Char)
  //class NRE extends RegExpTree

  implicit lazy val arbSymbol: Arbitrary[MyChar] = Arbitrary(oneOf(new MyChar('a'), new MyChar('b'), new MyChar('c'), new MyChar('d')))
  implicit lazy val arbNodeType: Arbitrary[NodeType] = Arbitrary(oneOf(SYMBOL, OPERATOR))
  implicit lazy val arbOperator: Arbitrary[OperatorType] = Arbitrary(oneOf(CONCAT, UNION, ITER))
  val maxM = 2
  implicit lazy val arbOrder: Arbitrary[Int] = Arbitrary(choose(1, maxM))

  implicit def arbRegExp(implicit s: Arbitrary[MyChar]): Arbitrary[RegExpTree] =
    Arbitrary {

      val genSymbol = for (e <- s.arbitrary) yield SymbolNode(e.c.toString)
        def genIter(sz: Int): Gen[RegExpTree] = for (r <- sizedTree(sz / 2)) yield OperatorNode(ITER, List(r))
        def genConcat(sz: Int): Gen[RegExpTree] = for {
          r1 <- sizedTree(sz / 2)
          r2 <- sizedTree(sz / 2)
        } yield OperatorNode(CONCAT, List(r1, r2))
        def genUnion(sz: Int): Gen[RegExpTree] = for {
          r1 <- sizedTree(sz / 2)
          r2 <- sizedTree(sz / 2)
        } yield OperatorNode(UNION, List(r1, r2))

        /*def genOper(sz: Int): Gen[RegExpTree] = for {
        n <- Arbitrary.arbitrary[NodeType]
        o <- Arbitrary.arbitrary[OperatorType]
        y <- s.arbitrary
      } yield ( if (n==SYMBOL) SymbolNode(y.c.toString)
                else {
                  if (o == ITER) OperatorNode(o, List(sizedTree(sz/2)))
                }
        )*/

        def sizedTree(sz: Int) =
          if (sz <= 0) genSymbol
          else Gen.frequency((1, genSymbol), (1, genConcat(sz)), (1, genUnion(sz)), (1, genIter(sz))) //suchThat(RegExpUtils.leastSymbolsNo(_) != 0)

      Gen.sized(sz => sizedTree(sz))
    }

  val normalRE = Arbitrary.arbitrary[RegExpTree] suchThat (RegExpUtils.leastSymbolsNo(_) != 0)

  //implicit def arbNormalTree(): Arbitrary[NRE] =
  //  Arbitrary {
  //    for (r <- normalRE) yield r.asInstanceOf[NRE]
  //  }

  /*val myProp = property("test") = forAll { (re: RegExpTree, m: Int) =>
    println("\n\n\tTesting pattern: " + re.toString)
    val policy = NONOVERLAP
    println(m)
    val es = StreamFactory.getCSVStreamSource(1000, Map('a' -> 0.25, 'b' -> 0.25, 'c' -> 0.25, 'd' -> 0.25), 10)
    val dfaProvider = DFAProvider(new DFASourceRegExp(re, policy, 0, es.getEventTypes()))
    val erf = ERFTask(dfaProvider, es)
    val prof0 = erf.execute()
    val dfaProviderM = DFAProvider(new DFASourceRegExp(re, policy, m, es.getEventTypes()))
    val erfM = ERFTask(dfaProviderM, es)
    val profM = erfM.execute()

    prof0.getMatchDump().checkAgainst(profM.getMatchDump())

  }*/

  val myProp2 = property("test2") = Prop.forAll(normalRE) { re =>
    println("\n\n\tTesting pattern: " + re.toString)
    val policy = ConfigUtils.defaultPolicy
    val m: Int = Gen.choose(1, maxM).sample.get
    println(m)
    val ss = StreamFactory.getStreamSource(1000, Map("a" -> 0.25, "b" -> 0.25, "c" -> 0.25, "d" -> 0.25), 10)
    val es = ss.emitEventsAndClose(EmitMode.BUFFER)
    val dfaProvider = DFAProvider(DFASourceRegExp(re, policy, 0, es.getEventTypes))
    val fsmp = FSMProvider(dfaProvider)
    val erf = ERFTask(fsmp, ss)
    val prof0 = erf.execute()
    val dfaProviderM = DFAProvider(DFASourceRegExp(re, policy, m, es.getEventTypes))
    val fsmpM = FSMProvider(dfaProviderM)
    val erfM = ERFTask(fsmpM, ss)
    val profM = erfM.execute()

    prof0.getMatchDump.checkAgainst(profM.getMatchDump)
    //}
    /*for (m <- 1 to maxM) {
      val dfaProvider = DFAProvider(new DFASourceRegExp(re,policy,m,es.getEventTypes()))
      val erf = ERFTask(dfaProvider,es)
      val profm = erf.execute()
      DFAUtils.isMUnambiguous(erf.getEngine.getDFA(),m)
      prof0.getMatchDump().checkAgainst(profm.getMatchDump())
    }*/
    //(0 == 0)
  }

}
