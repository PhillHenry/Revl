package uk.co.odinconsultants.revl.wikipedia

import com.henryp.sparkfinance.config.Spark
import com.henryp.sparkfinance.logging.Logging
import org.apache.commons.io.FileUtils
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.{SparseMatrix, Vector, Vectors}
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
  val originalRows    = config.k * 3
  val originalColumns = config.k * 2

  "SVD" should {
    "generate 3 matrices" in {
      val indexedRowMatrix  = generateMatrix(originalRows, config)
      val svd               = computeSVD(config, indexedRowMatrix)
      svd.s.size shouldEqual config.k
      svd.U.rows.count() shouldEqual originalRows
      svd.V.numRows shouldEqual originalColumns
      svd.V.numCols shouldEqual config.k

      val q = createArbitraryQuery
      newQuery(q, svd.U, svd.s)

      WikipediaMain.save(config, sc, svd)

      val uFromFileRDD  = loadUFromFile
      val sFromFile     = loadSFromFile
      val newQ          = newQuery(q, new IndexedRowMatrix(uFromFileRDD), sFromFile)

      val similaritiesMatrix = newQ.toIndexedRowMatrix().toRowMatrix().multiply(svd.V.transpose).columnSimilarities()
      similaritiesMatrix.numCols() shouldEqual originalColumns
    }
  }

  def loadSFromFile: Vector = {
    Vectors.dense(Source.fromFile(toDir(config.singularValuesFilename) + "/part-00000").toArray.map(_.toDouble))
  }

  def loadUFromFile: RDD[IndexedRow] = {
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

  override protected def afterAll(): Unit = {
    sc.stop()
  }
}
