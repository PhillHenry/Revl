package uk.co.odinconsultants.revl

import com.henryp.sparkfinance.config.Spark
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

package object wikipedia {
  case class WikipediaConfig(directory: String     = "/home/henryp/Mounts/CorsairData",
                             saveDirectory: String = "hdfs://192.168.1.15:8020/Wikipedia/",
                             sparkUrl: String      = Spark.localMaster,
                             k: Int                = 100,
                             jars: Seq[String]     = List[String](),
                             stopwordsFile: String = "/home/henryp/Code/Scala/Spark/AdvancedAnalyticsWithSpark/./aas/ch06-lsa/src/main/resources/stopwords.txt",
                             numPartitions: Int    = 2) {
    val rightSingularFilename   = s"$saveDirectory/right_singular_vectors.rdd"
    val leftSingularFilename    = s"$saveDirectory/left_singular_vectors.rdd"
    val singularValuesFilename  = s"$saveDirectory/singular_values.rdd"
  }

  def parseArgs(args: Array[String]): Option[WikipediaConfig] = {
    val parser = new scopt.OptionParser[WikipediaConfig]("Wikipedia") {
      opt[String]('d', "directory") action { case(value, config) => config.copy(directory = value) } text "data directory"
      opt[String]('s', "spark") action { case(value, config) => config.copy(sparkUrl = value) } text "spark URL"
      opt[String]('w', "stopwordsFile") action { case(value, config) => config.copy(stopwordsFile = value) } text "file containing stopwords"
      opt[String]('o', "output") action { case(value, config) => config.copy(saveDirectory = value) } text "output directory"
      opt[Seq[String]]('j', "jars") valueName "<jar1>,<jar2>..."  action { (value, config) =>
        config.copy(jars = value)
      } text "jars"
      opt[Int]('k', "k") action { case(value, config) => config.copy(k = value) } text "k value for SVM"
      opt[Int]('p', "partitions") action { case(value, config) => config.copy(numPartitions = value) } text "number of partions"
    }
    parser.parse(args, WikipediaConfig())
  }

  def getSparkContext(config: WikipediaConfig): SparkContext = {
    val context = Spark.sparkContext(config.sparkUrl)
    config.jars.foreach { jar =>
      println(s"Adding JAR $jar")
      context.addJar(jar)
    }
    context
  }

  def save(rdd: RDD[_], filename: String): Unit = rdd.saveAsTextFile(filename)

}
