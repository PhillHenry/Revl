package uk.co.odinconsultants.revl.wikipedia

import java.util.Date

import com.henryp.sparkfinance.logging.Logging
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import org.apache.spark.rdd.RDD

import scala.util.{Failure, Success, Try}

/**
  * Taken from chapter 6 of Advanced Analytics With Spark
  *
  * Run with app arguments something like:
  *
-s spark://192.168.1.9:7077 -o hdfs://192.168.1.15:8020/Wikipedia/3Jan16 -p 20 -j /home/henryp/Code/Scala/Revl/wikipedia/target/wikipedia-1.0-SNAPSHOT.jar
  *
  *
  */
object WikipediaMain extends Logging {

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
    val tryProcess    = Try {
      val (termDocRDD, termIds, docIds, idfs)
                        = analytics.preprocessing(sampleSize = 0.2, numTerms = 50000)
      val termDocMatrix = new IndexedRowMatrix(termDocRDD.zipWithUniqueId().map(x => IndexedRow(x._2, x._1)))
      val svd           = computeSVD(config, termDocMatrix)

      save(config, sc, svd)
    }

    tryProcess match {
      case Success(_) => println("done")
      case Failure(x) => x.printStackTrace()
    }

    info("Press any key. (" + new Date() + ")")
    Console.in.readLine()
  }

  /**
    * @see see https://gist.github.com/vrilleup/9e0613175fab101ac7cd
    */
  def save(config: WikipediaConfig, sc: SparkContext, svd: SingularValueDecomposition[IndexedRowMatrix, Matrix]): Unit = {
    save(svd.U.rows, config.leftSingularFilename)
    val V = svd.V.toArray.grouped(svd.V.numRows).toList.transpose
    sc.makeRDD(V, 1).zipWithIndex().map(_.swap)
      .map(toTSV) // make tsv line starting with column index
      .saveAsTextFile(config.rightSingularFilename)
    sc.makeRDD(svd.s.toArray, 1).saveAsTextFile(config.singularValuesFilename)
  }

  def computeSVD(config: WikipediaConfig, termDocMatrix: IndexedRowMatrix): SingularValueDecomposition[IndexedRowMatrix, Matrix] = {
    termDocMatrix.computeSVD(k = config.k, computeU = true)
  }

  def save(rdd: RDD[IndexedRow], filename: String): Unit = {
    rdd.map(row => (row.index, row.vector.toArray.toList)).map(toTSV).saveAsTextFile(filename)
  }

  def toTSV: ((Long, TraversableOnce[Double])) => String = {
    line => line._1 + delimiter + line._2.mkString(delimiter)
  }
}
