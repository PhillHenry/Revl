package uk.co.odinconsultants.revl.wikipedia

import com.cloudera.datascience.lsa.ParseWikipedia
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.{Map, mutable}

class AdvancedAnalyticsWithSparkCh6(config: WikipediaConfig, sc: SparkContext) {

  def preprocessing(sampleSize: Double, numTerms: Int): (RDD[Vector], Map[Int, String], Map[Long, String], Map[String, Double]) = {
    val pages       = ParseWikipedia.readFile(config.directory, sc).sample(false, sampleSize, 11L)
    val plainText   = pages.filter(_ != null).flatMap(ParseWikipedia.wikiXmlToPlainText(_))
    val stopWords   = sc.broadcast(ParseWikipedia.loadStopWords(config.stopwordsFile)).value

    val lemmatized = plainText.mapPartitions(iter => {
      val pipeline = ParseWikipedia.createNLPPipeline()
      iter.map{ case(title, contents) => (title, ParseWikipedia.plainTextToLemmas(contents, stopWords, pipeline))}
    })

    val filtered = lemmatized.filter(_._2.size > 1)

    documentTermMatrix(filtered, stopWords, numTerms)
  }

  /**
    * Returns a document-term matrix where each element is the TF-IDF of the row's document and
    * the column's term.
    */
  def documentTermMatrix(docs:      RDD[(String, Seq[String])],
                         stopWords: Set[String],
                         numTerms:  Int): (RDD[Vector], Map[Int, String], Map[Long, String], Map[String, Double]) = {
    val docTermFreqs = docToTermFreq(docs)
    val docIds       = idsToDocs(docTermFreqs)
    val docFreqs     = ParseWikipedia.documentFrequenciesDistributed(docTermFreqs.map(_._2), numTerms)
    println("Number of terms: " + docFreqs.size)
    ParseWikipedia.saveDocFreqs("docfreqs.tsv", docFreqs)

    val numDocs     = docIds.size
    val idfs        = ParseWikipedia.inverseDocumentFrequencies(docFreqs, numDocs)
    // Maps terms to their indices in the vector
    val idTerms     = idfs.keys.zipWithIndex.toMap
    val termIds     = idTerms.map(_.swap)

    val vecs        = scores(docTermFreqs, idfs, idTerms)
    (vecs, termIds, docIds, idfs)
  }

  def scores(docTermFreqs: RDD[(String, mutable.HashMap[String, Int])], idfs: Map[String, Double], idTerms: Predef.Map[String, Int]): RDD[Vector] = {
    val bIdfs = sc.broadcast(idfs).value
    val bIdTerms = sc.broadcast(idTerms).value

    val vecs = docTermFreqs.map(_._2).map(termFreqs => {
      val docTotalTerms = termFreqs.values.sum
      val termScores = termFreqs.filter {
        case (term, freq) => bIdTerms.contains(term)
      }.map {
        case (term, freq) =>
          val id = bIdTerms(term)
          (id, bIdfs(term) * termFreqs(term) / docTotalTerms)
      }.toSeq
      val vec = Vectors.sparse(bIdTerms.size, termScores)
      vec
    })
    vecs.cache()
    save(vecs, s"${config.saveDirectory}/termDocMatrix.rdd")
    vecs
  }

  def idsToDocs(docTermFreqs: RDD[(String, mutable.HashMap[String, Int])]): Map[Long, String] = {
    val docIdsRdd = docTermFreqs.map(_._1).zipWithUniqueId().map(_.swap)
    val docIds    = docIdsRdd.collectAsMap()
    println("saving docIdsRdd")
    save(docIdsRdd, s"${config.saveDirectory}/docIdsRdd.rdd")
    docIds
  }

  def docToTermFreq(docs: RDD[(String, Seq[String])]): RDD[(String, mutable.HashMap[String, Int])] = {
    val docTermFreqs = docs.mapValues(terms => {
      val termFreqsInDoc = terms.foldLeft(new mutable.HashMap[String, Int]()) {
        (map, term) => map += term -> (map.getOrElse(term, 0) + 1)
      }
      termFreqsInDoc
    })
    docTermFreqs.cache()
    docTermFreqs
  }


}
