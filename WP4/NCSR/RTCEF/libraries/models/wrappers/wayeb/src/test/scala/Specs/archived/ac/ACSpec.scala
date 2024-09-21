package Specs.archived.ac

import fsm.archived.AhoCorasick.ACFactory
import org.scalatest.FlatSpec
import fsm.CountPolicy._
import scala.collection.mutable.Set

class ACSpec extends FlatSpec {
  "Patterns " should " produce correct DFAs " in {

    testSimple()
    testWithExtra1()
    testWithExtra2()
    testWithACExample()

  }

  it should "throw IllegalArgumentException if (policy not (0 or 1)) or no keywords provided" in {
    assertThrows[IllegalArgumentException] {
      val aca = ACFactory.buildACA(List(), OVERLAP, Set())
    }
  }

  def testSimple(): Unit = {
    println("Testing simple pattern ab...")
    val aca1 = ACFactory.buildACA(List("ab"), OVERLAP, Set())
    println("ACA\n" + aca1.toString() + "\nDFA\n" + aca1.dfaAsString())
    assert(aca1.getStates.size == 3)
    println("Testing DFA...")
    assert(aca1.getStates(0).getDelta('a') == 1)
    assert(aca1.getStates(0).getDelta('b') == 0)
    assert(aca1.getStates(1).getDelta('a') == 1)
    assert(aca1.getStates(1).getDelta('b') == 2)
    assert(aca1.getStates(2).getDelta('a') == 1)
    assert(aca1.getStates(2).getDelta('b') == 0)
  }

  def testWithExtra1(): Unit = {
    println("Testing simple pattern ab with extra symbol(s) not present in pattern ('c')...")
    val aca2 = ACFactory.buildACA(List("ab"), OVERLAP, Set('c'))
    println("ACA\n" + aca2.toString() + "\nDFA\n" + aca2.dfaAsString())
    assert(aca2.getStates.size == 3)
    println("Testing DFA...")
    assert(aca2.getStates(0).getDelta('a') == 1)
    assert(aca2.getStates(0).getDelta('b') == 0)
    assert(aca2.getStates(0).getDelta('c') == 0)
    assert(aca2.getStates(1).getDelta('a') == 1)
    assert(aca2.getStates(1).getDelta('b') == 2)
    assert(aca2.getStates(1).getDelta('c') == 0)
    assert(aca2.getStates(2).getDelta('a') == 1)
    assert(aca2.getStates(2).getDelta('b') == 0)
    assert(aca2.getStates(2).getDelta('c') == 0)
  }

  def testWithExtra2(): Unit = {
    println("Testing simple pattern ab with extra symbol(s) both present and not present in pattern ('b','c')...")
    val aca3 = ACFactory.buildACA(List("ab"), OVERLAP, Set('b', 'c'))
    println("ACA\n" + aca3.toString() + "\nDFA\n" + aca3.dfaAsString())
    assert(aca3.getStates.size == 3)
    println("Testing DFA...")
    assert(aca3.getStates(0).getDelta('a') == 1)
    assert(aca3.getStates(0).getDelta('b') == 0)
    assert(aca3.getStates(0).getDelta('c') == 0)
    assert(aca3.getStates(1).getDelta('a') == 1)
    assert(aca3.getStates(1).getDelta('b') == 2)
    assert(aca3.getStates(1).getDelta('c') == 0)
    assert(aca3.getStates(2).getDelta('a') == 1)
    assert(aca3.getStates(2).getDelta('b') == 0)
    assert(aca3.getStates(2).getDelta('c') == 0)
  }

  def testWithACExample(): Unit = {
    println("Testing with the example in the Aho-Corasick paper...")
    val aca4 = ACFactory.buildACA(List("he", "she", "his", "hers"), OVERLAP, Set('u'))
    println("ACA\n" + aca4.toString() + "\nDFA\n" + aca4.dfaAsString())
    assert(aca4.getStates.size == 10)
    println("Testing DFA...")
    // ('e','h','i','r','s','u')
    assert(aca4.getStates(0).getDelta('e') == 0)
    assert(aca4.getStates(0).getDelta('h') == 1)
    assert(aca4.getStates(0).getDelta('i') == 0)
    assert(aca4.getStates(0).getDelta('r') == 0)
    assert(aca4.getStates(0).getDelta('s') == 3)
    assert(aca4.getStates(0).getDelta('u') == 0)

    assert(aca4.getStates(1).getDelta('e') == 2)
    assert(aca4.getStates(1).getDelta('h') == 1)
    assert(aca4.getStates(1).getDelta('i') == 6)
    assert(aca4.getStates(1).getDelta('r') == 0)
    assert(aca4.getStates(1).getDelta('s') == 3)
    assert(aca4.getStates(1).getDelta('u') == 0)

    assert(aca4.getStates(2).getDelta('e') == 0)
    assert(aca4.getStates(2).getDelta('h') == 1)
    assert(aca4.getStates(2).getDelta('i') == 0)
    assert(aca4.getStates(2).getDelta('r') == 8)
    assert(aca4.getStates(2).getDelta('s') == 3)
    assert(aca4.getStates(2).getDelta('u') == 0)

    assert(aca4.getStates(3).getDelta('e') == 0)
    assert(aca4.getStates(3).getDelta('h') == 4)
    assert(aca4.getStates(3).getDelta('i') == 0)
    assert(aca4.getStates(3).getDelta('r') == 0)
    assert(aca4.getStates(3).getDelta('s') == 3)
    assert(aca4.getStates(3).getDelta('u') == 0)

    assert(aca4.getStates(4).getDelta('e') == 5)
    assert(aca4.getStates(4).getDelta('h') == 1)
    assert(aca4.getStates(4).getDelta('i') == 6)
    assert(aca4.getStates(4).getDelta('r') == 0)
    assert(aca4.getStates(4).getDelta('s') == 3)
    assert(aca4.getStates(4).getDelta('u') == 0)

    assert(aca4.getStates(5).getDelta('e') == 0)
    assert(aca4.getStates(5).getDelta('h') == 1)
    assert(aca4.getStates(5).getDelta('i') == 0)
    assert(aca4.getStates(5).getDelta('r') == 8)
    assert(aca4.getStates(5).getDelta('s') == 3)
    assert(aca4.getStates(5).getDelta('u') == 0)

    assert(aca4.getStates(6).getDelta('e') == 0)
    assert(aca4.getStates(6).getDelta('h') == 1)
    assert(aca4.getStates(6).getDelta('i') == 0)
    assert(aca4.getStates(6).getDelta('r') == 0)
    assert(aca4.getStates(6).getDelta('s') == 7)
    assert(aca4.getStates(6).getDelta('u') == 0)

    assert(aca4.getStates(7).getDelta('e') == 0)
    assert(aca4.getStates(7).getDelta('h') == 4)
    assert(aca4.getStates(7).getDelta('i') == 0)
    assert(aca4.getStates(7).getDelta('r') == 0)
    assert(aca4.getStates(7).getDelta('s') == 3)
    assert(aca4.getStates(7).getDelta('u') == 0)

    assert(aca4.getStates(8).getDelta('e') == 0)
    assert(aca4.getStates(8).getDelta('h') == 1)
    assert(aca4.getStates(8).getDelta('i') == 0)
    assert(aca4.getStates(8).getDelta('r') == 0)
    assert(aca4.getStates(8).getDelta('s') == 9)
    assert(aca4.getStates(8).getDelta('u') == 0)

    assert(aca4.getStates(9).getDelta('e') == 0)
    assert(aca4.getStates(9).getDelta('h') == 4)
    assert(aca4.getStates(9).getDelta('i') == 0)
    assert(aca4.getStates(9).getDelta('r') == 0)
    assert(aca4.getStates(9).getDelta('s') == 3)
    assert(aca4.getStates(9).getDelta('u') == 0)
  }

}
