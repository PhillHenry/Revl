package uk.co.odinconsultants.revl.wikipedia

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import org.apache.spark.rdd.RDD

/**
  * Taken from chapter 6 of Advanced Analytics With Spark
  */
object WikipediaMain {

  val delimiter = "\t"

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

  def computeSVD(config: WikipediaConfig, termDocMatrix: IndexedRowMatrix): SingularValueDecomposition[IndexedRowMatrix, Matrix] = {
    termDocMatrix.computeSVD(k = config.k, computeU = true)
  }

  def save(rdd: RDD[IndexedRow], filename: String): Unit = {
    rdd.map(row => (row.index, row.vector.toArray)).map(line => line._1 + delimiter + line._2.mkString(delimiter)).saveAsTextFile(filename)
  }


}
