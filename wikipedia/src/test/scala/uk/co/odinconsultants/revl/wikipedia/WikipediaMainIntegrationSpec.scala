package uk.co.odinconsultants.revl.wikipedia

import com.henryp.sparkfinance.config.Spark
import com.henryp.sparkfinance.logging.Logging
import org.apache.commons.io.FileUtils
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.{Matrix, SparseMatrix, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import uk.co.odinconsultants.revl.wikipedia.MatrixOperations._
import uk.co.odinconsultants.revl.wikipedia.WikipediaMain._

import scala.io.Source
import scala.util.Random

class WikipediaMainIntegrationSpec extends WordSpec with Matchers with BeforeAndAfterAll with Logging {

  val sc              = Spark.sparkContext()
  val url             = this.getClass.getResource("/") + "../../target/test-rdds/"
  val dir             = toDir(url)
  info(s"Deleting $dir")
  FileUtils.deleteQuietly(new java.io.File(dir))
  val config          = WikipediaConfig(k = 100, saveDirectory = url)
  val numDocs         = config.k * 3
  val numTerms        = config.k * 2

  /**
    * Beware! "Return statements aren't allowed in Spark closures"
    * @see http://stackoverflow.com/questions/27782923/compare-data-in-two-rdd-in-spark
    */
  def compare(newQ: BlockMatrix, decomposedQ: BlockMatrix): Boolean = {
    val zero: Option[(Option[Matrix], Option[Matrix])] = Some(Some(null.asInstanceOf[Matrix]), Some(null.asInstanceOf[Matrix]))
    val joined = newQ.blocks.fullOuterJoin(decomposedQ.blocks).map(x => Option(x._2)).fold(zero) { case (acc, pair) =>
//      acc.map { ignored =>
//        ???
//      }
      None
    }
    true
  }

//  def compare(kv: (Option[Matrix], Option[Matrix])): Option[Matrix] = {
//    val left = kv._1
//    val right = kv._2
//    left match {
//      case None =>
//        right match {
//          case Some(_) => None
//        }
//      case Some(leftMatrix) =>
//        right match {
//          case None => None
//          case Some(rightMatrix) =>
//            (0 to leftMatrix.toArray.size).foreach { index =>
//              if (leftMatrix.toArray(index) != rightMatrix.toArray(index)) None
//              Some(null.asInstanceOf[Matrix])
//            }
//        }
//    }
//  }

  "SVD" should {
    "generate 3 matrices" in {
      val indexedRowMatrix  = generateMatrix(numDocs, numTerms)
      val svd               = computeSVD(config, indexedRowMatrix)
      // remember: X = U D V^T^
      svd.s.size shouldEqual config.k
      svd.U.rows.count() shouldEqual numDocs
      svd.V.numRows shouldEqual numTerms
      svd.V.numCols shouldEqual config.k

      val q           = createArbitraryQuery
      val decomposedQ = newQuery(q, svd.U, svd.s)

      WikipediaMain.save(config, sc, svd)

      val uFromFileRDD  = loadUFromFile()
      val sFromFile     = loadSFromFile()
      val uRowMatrix    = new IndexedRowMatrix(uFromFileRDD)
      val newQ          = newQuery(q, uRowMatrix, sFromFile)
//      decomposedQ shouldEqual newQ // No, you'll have to be smarter at comparing matrices
      compare(newQ, decomposedQ)

      // V's columns are called the right singular vectors. Each row corresponds to a term and each column corresponds to a concept
      val similaritiesMatrix = newQ.toIndexedRowMatrix().toRowMatrix().multiply(svd.V.transpose).columnSimilarities()
      similaritiesMatrix.numCols() shouldEqual numTerms
      // see http://stackoverflow.com/questions/29860472/spark-how-can-i-retrieve-item-pair-after-calculating-similarity-using-rowmatrix
      // particularly .columnSimilarities().entries() comment

      checkOrthogonal(uRowMatrix.toBlockMatrix())
    }
  }

  def checkOrthogonal(blockMatrix: BlockMatrix): Unit = {
    val beWithinTolerance = be >= 0d and be <= 1e-8
    val uxuT      = blockMatrix.transpose.multiply(blockMatrix)
    val uxuTLocal = uxuT.toLocalMatrix()
    for (i <- 0 until uxuT.numRows().toInt) {
      for (j <- 0 until uxuT.numCols().toInt) {
        withClue(s"i = $i, j = $j") {
          val element = uxuTLocal(i, j)
//          println(s"singular matrix element for ($i, $j) = $element")
          if (i != j) {
            Math.abs(element) should beWithinTolerance
          }
        }
      }
    }
  }

  def loadSFromFile(): Vector = {
    Vectors.dense(Source.fromFile(toDir(config.singularValuesFilename) + "/part-00000").toArray.map(_.toDouble))
  }

  def loadUFromFile(): RDD[IndexedRow] = {
    val uFromFileRDD = sc.textFile(config.leftSingularFilename).map { line =>
      val elements = line.split(delimiter)
      IndexedRow(elements(0).toInt, Vectors.dense(elements.drop(1).map(_.toDouble)))
    }
    uFromFileRDD
  }

  def toDir(url: String): String = url.replaceFirst("^file:", "")

  def newQuery(q: SparseMatrix, u: IndexedRowMatrix, s: Vector): BlockMatrix = {
    val newQ = createNewQuery(q, u, s, sc)
    newQ.validate()
    newQ.numRows() shouldEqual 1
    newQ.numCols() shouldEqual config.k
    newQ
  }


  def createArbitraryQuery: SparseMatrix = {
    val indices = Array(0, numDocs - 1)
    val values = Array(1d, 10d)
    new SparseMatrix(1, numDocs, Array(0, indices.length), indices, values)
  }

  def generateMatrix(rows: Int, cols: Int): IndexedRowMatrix = {
    val indexedRows       = toIndexedRows(generateRddOfRandomVectors(rows, cols))
    new IndexedRowMatrix(indexedRows)
  }

  def toIndexedRows(rdd: RDD[Vector]): RDD[IndexedRow] = {
    rdd.zipWithUniqueId().map(x => IndexedRow(x._2, x._1))
  }

  def generateRddOfRandomVectors(rows: Int, cols: Int): RDD[Vector] = {
    val rdd = sc.makeRDD((1 to rows).map(n => Vectors.dense(randomDoubles(cols))))
    rdd.cache()
  }

  def randomDoubles(n: Int): Array[Double] = {
    val array = Array.fill(n)(0d)

    for (i <- 0 to (n / 10)) {
      array(Random.nextInt(n)) = Random.nextDouble()
    }

    array
  }

  override protected def afterAll(): Unit = {
    sc.stop()
  }
}
