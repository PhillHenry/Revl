package uk.co.odinconsultants.revl.wikipedia

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.{DenseVector, Matrix, SingularValueDecomposition, SparseMatrix}
import org.apache.spark.rdd.RDD

/**
  * Taken from chapter 6 of Advanced Analytics With Spark
  */
object WikipediaMain {

  def main(args: Array[String]): Unit = {
    val configOpt = parseArgs(args)
    configOpt foreach { config =>
      run(config)
    }
  }

  def run(config: WikipediaConfig): Unit = {
    val sc            = getSparkContext(config)
    val analytics     = new AdvancedAnalyticsWithSparkCh6(config, sc)
    val (termDocRDD, termIds, docIds, idfs)
                      = analytics.preprocessing(sampleSize = 0.1, numTerms = 50000)
    val termDocMatrix = new IndexedRowMatrix(termDocRDD.zipWithUniqueId().map(x => IndexedRow(x._2, x._1)))
    val svd           = computeSVD(config, termDocMatrix)

    save(config, sc, svd)
  }

  val delimiter = "\t"

  /**
    * @see see https://gist.github.com/vrilleup/9e0613175fab101ac7cd
    */
  def save(config: WikipediaConfig, sc: SparkContext, svd: SingularValueDecomposition[IndexedRowMatrix, Matrix]): Unit = {
    save(svd.U.rows, config.leftSingularFilename)
    val V = svd.V.toArray.grouped(svd.V.numRows).toList.transpose
    sc.makeRDD(V, 1).zipWithIndex()
      .map(line => line._2 + delimiter + line._1.mkString(delimiter)) // make tsv line starting with column index
      .saveAsTextFile(config.rightSingularFilename)
    sc.makeRDD(svd.s.toArray, 1).saveAsTextFile(config.singularValuesFilename)
  }

  /**
    * q U = ( ( q U )T )T = ( UT qT)T
    *
    * @see http://www1.se.cuhk.edu.hk/~seem5680/lecture/LSI-Eg.pdf
    */
  def q_x_U(q: SparseMatrix, u: IndexedRowMatrix, sc: SparkContext): BlockMatrix = {
    val qIndexedRows  = toSeq(q)
    val qRdd          = sc.makeRDD(qIndexedRows)
    val qIndexed      = new IndexedRowMatrix(qRdd, q.numCols, q.numRows)
    val qBlock        = qIndexed.toBlockMatrix()
    u.toBlockMatrix().transpose.multiply(qBlock).transpose
  }

  def toSeq(q: SparseMatrix): Seq[IndexedRow] = {
    var i = 0
    q.rowIndices.map { index =>
      val row = IndexedRow(index, new DenseVector(Array(q.values(i))))
      i += 1
      row
    }
  }

  def computeSVD(config: WikipediaConfig, termDocMatrix: IndexedRowMatrix): SingularValueDecomposition[IndexedRowMatrix, Matrix] = {
    termDocMatrix.computeSVD(k = config.k, computeU = true)
  }

  def save(rdd: RDD[IndexedRow], filename: String): Unit = {
    rdd.map(row => (row.index, row.vector.toArray)).map(line => line._1 + delimiter + line._2.mkString(delimiter)).saveAsTextFile(filename)
  }


}
