package scripts.sessionsNtests.smile

import java.io.FileInputStream

import smile.classification.{SVM, knn}
import smile.data.{AttributeDataset, NominalAttribute}
import smile.data.parser.DelimitedTextParser
import smile.math.kernel.GaussianKernel
import smile.read
import smile.regression.ols
import smile.sequence.HMM

object tutorial {
  val smileHome = System.getenv("HOME") + "/opt/ml/smile-1.5.3"

  def main(args: Array[String]): Unit = {
    //leastSquaresTest
    //knnTest
    hmmTest
  }

  def svmTest: Unit = {
    val parser: DelimitedTextParser = new DelimitedTextParser()
    parser.setResponseIndex(new NominalAttribute("class"), 0)
    try {
      val train: AttributeDataset = parser.parse("USPS Train", new FileInputStream(smileHome + "/scripts/data/usps/zip.train"))
      val test: AttributeDataset = parser.parse("USPS Train", new FileInputStream(smileHome + "/scripts/data/usps/zip.test"))

      val dd = new Array[Array[Double]](train.size())
      val x = train.toArray(dd)
      val y = train.toArray(new Array[Int](train.size()))
      val testx = test.toArray(new Array[Array[Double]](test.size()))
      val testy = train.toArray(new Array[Int](test.size()))

      val svm = new SVM[Array[Double]](new GaussianKernel(8.0), 5.0, y.max + 1, SVM.Multiclass.ONE_VS_ONE)
      svm.learn(x, y)
      svm.finish()

      var error = 0
      for (i <- 0 until testx.length) {
        if (svm.predict(testx(i)) != testy(i)) error += 1
      }
      println("USPS error rate = " + 100.0 * error / testx.length)

    } catch {
      case e: Exception => println(e)
    }
  }

  def leastSquaresTest: Unit = {
    val planes = read.arff(smileHome + "/scripts/data/weka/regression/2dplanes.arff", 10)
    val (x, y): (Array[Array[Double]], Array[Double]) = planes.unzipDouble
    val model = ols(x, y)
    println(model)
  }

  def knnTest: Unit = {
    val iris = read.arff(smileHome + "/scripts/data/weka/iris.arff", 4)
    val (x, y): (Array[Array[Double]], Array[Int]) = iris.unzipInt
    val model = knn(x, y, 3)
    println(model)
  }

  def hmmTest: Unit = {
    //val x: Array[Array[Int]] = Array(Array(0),Array(0),Array(0),Array(1),Array(1),Array(1),Array(2),Array(2),Array(2))
    //val y: Array[Array[Int]] = Array(Array(0),Array(0),Array(0),Array(1),Array(1),Array(1),Array(0),Array(0),Array(0))
    val x: Array[Array[Int]] = Array(Array(0, 0, 0, 1, 1, 1, 2, 2, 2))
    val y: Array[Array[Int]] = Array(Array(0, 0, 0, 1, 1, 1, 0, 0, 0))
    val model = new HMM[Int](x, y)
    println(model)
    val obs1: Array[Int] = Array(0, 0)
    val obs1p = model.p(obs1)
    println("obs1 " + obs1p)
    val pred1 = model.predict(obs1)
    println("pred1 " + pred1)
  }

}
