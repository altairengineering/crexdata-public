import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, argmin, max, min, sum}

object breezetests {
  def mypartition(inds: List[(Double,Int)], vals: List[Double]): List[List[Int]] = {
    vals match {
      case Nil => Nil
      case _ => inds.filter(x => x._1==vals.head).map(x => x._2) :: mypartition(inds,vals.tail)
    }
  }

  /*def myrankAux(ll: List[List[Int]], currentRank: Int): Map[Int,List[Int]] = {

  }*/

  def myrank(ll: List[List[Int]]): List[(Int,List[Int])] = {
    val rank = ll.flatten.size
    ll match {
      case Nil => Nil
      case head::tail => (rank,head) :: myrank(tail)
    }
  }

  def myrank2Aux(ll: List[List[Int]], seen: Int): List[(Int,List[Int])] = {
    require(seen >= 0)
    ll match {
      case Nil => Nil
      case head::tail => (seen+1,head) :: myrank2Aux(tail,seen+head.size)
    }
  }

  def myrank2(ll: List[List[Int]]): List[(Int,List[Int])] = {
    myrank2Aux(ll,0)
  }

  class Rank(r: Map[Int,List[Int]]) {
    def getRanking = r
    def getRankFor(j: Int): Int = {
      getRankForAux(j,r.toList)
    }
    private def getRankForAux(j: Int, rl: List[(Int,List[Int])]): Int = {
      rl match {
        case Nil => throw new IllegalArgumentException
        case head::tail => if (head._2.contains(j)) head._1 else getRankForAux(j,tail)
      }
    }
    override def toString: String = r.toString()
  }



  val m = DenseMatrix.zeros[Double](4,4)
  m(0,0) = 0.001
  m(1,1) = 11

  m(2,0) = 0.0
  m(2,1) = 0.0
  m(2,2) = 2.2
  m(2,3) = 2.1

  m(3,3) = 0.33
  //m
  val tmp = m(2,::).t

  max(tmp)
  val minel = min(tmp)
  //argmax(tmp)
  //argmin(tmp)

  minel

  val tmpl = tmp.toArray.toList
  val tmpi = tmpl.zipWithIndex

  tmpl.distinct.sorted

  tmpi.filter( x => x._1==minel).map(x => x._2)

  tmp
  val par = mypartition(tmpi,tmpl.distinct.sorted)

  val r = new Rank(myrank(par).reverse.toMap)

  r.getRankFor(3)

  val lll = List(5,23,87,23,6,7)
  lll.indexOf(lll.max)

  val par2 = mypartition(tmpi,tmpl.distinct.sorted.reverse)
  val r2 = new Rank(myrank2(par2).toMap)

}