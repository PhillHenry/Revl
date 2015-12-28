package uk.co.odinconsultants.revl.wikipedia

import com.henryp.sparkfinance.config.Spark
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.{SparseMatrix, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import uk.co.odinconsultants.revl.wikipedia.WikipediaMain._

import scala.util.Random

class WikipediaMainIntegrationSpec extends WordSpec with Matchers with BeforeAndAfterAll {

  val sc              = Spark.sparkContext()
  val dir             = (this.getClass.getResource("/") + "../../src/test/resources/").replaceFirst("^file:", "file://")
  val config          = WikipediaConfig(k = 100, saveDirectory = dir)
  val originalRows    = config.k * 3
  val originalColumns = config.k * 2

  "SVD" should {
    "generate 3 matrices" in {
      val indexedRowMatrix  = generateMatrix(originalRows, config)
      val svd               = computeSVD(config, indexedRowMatrix)
      svd.s.size shouldEqual config.k
      svd.U.rows.count() shouldEqual originalRows
      svd.V.numRows shouldEqual originalColumns

      val q = createQuery
      val u = svd.U
      multiply(q, u)

      WikipediaMain.save(config, sc, svd)

      val uFromFileRDD = sc.textFile(config.leftSingularFilename).map { line =>
        val elements = line.split(delimiter)
        IndexedRow(elements(0).toInt, Vectors.dense(elements.drop(1).map(_.toDouble)))
      }
      multiply(q, new IndexedRowMatrix(uFromFileRDD))
    }
  }

  def multiply(q: SparseMatrix, u: IndexedRowMatrix): Unit = {
    val newQ = q_x_U(q, u, sc)
    newQ.validate()
    newQ.numRows() shouldEqual 1
    newQ.numCols() shouldEqual config.k
  }

  def createQuery: SparseMatrix = {
    val indices = Array(0, originalRows - 1)
    val values = Array(1d, 10d)
    new SparseMatrix(1, originalRows, Array(0, indices.length), indices, values)
  }

  def generateMatrix(rows: Int, config: WikipediaConfig): IndexedRowMatrix = {
    val indexedRows       = toIndexedRows(generateRddOfRandomVectors(rows, originalColumns))
    new IndexedRowMatrix(indexedRows)
  }

  def toIndexedRows(rdd: RDD[Vector]): RDD[IndexedRow] = {
    rdd.zipWithUniqueId().map(x => IndexedRow(x._2, x._1))
  }

  def generateRddOfRandomVectors(rows: Int, cols: Int): RDD[Vector] = {
    val rdd = sc.makeRDD((1 to rows).map(n => Vectors.dense(randomDoubles(cols))))
    rdd.cache()
    rdd
  }

  def randomDoubles(n: Int): Array[Double] = Array.fill(n)(Random.nextDouble())

//  override protected def beforeAll(): Unit = super.beforeAll()
  override protected def afterAll(): Unit = {
    sc.stop()
  }
}
