package uk.co.odinconsultants.revl.wikipedia

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg.{DenseVector, SparseMatrix, Vector, Vectors}

object MatrixOperations {

  def createNewQuery(q: SparseMatrix, u: IndexedRowMatrix, s: Vector, sc: SparkContext): BlockMatrix = {
    val u_x_SInv = multiplyByInverseDiagonalMatrix(u, s)
    multiply(q, u_x_SInv, sc)
  }

  /**
    * q U = ( ( q U )T )T = ( UT qT)T
    *
    * @see http://www1.se.cuhk.edu.hk/~seem5680/lecture/LSI-Eg.pdf
    */
  def multiply(q: SparseMatrix, u: IndexedRowMatrix, sc: SparkContext): BlockMatrix = {
    val qIndexedRows  = toSeq(q)
    val qRdd          = sc.makeRDD(qIndexedRows)
    val qTIndexed     = new IndexedRowMatrix(qRdd, q.numCols, q.numRows) // cols <-> rows => transpose
    val qTBlock       = qTIndexed.toBlockMatrix()
    u.toBlockMatrix().transpose.multiply(qTBlock).transpose
  }

  def toSeq(q: SparseMatrix): Seq[IndexedRow] = {
    var i = 0
    q.rowIndices.map { index =>
      val row = IndexedRow(index, new DenseVector(Array(q.values(i))))
      i += 1
      row
    }
  }

  def multiplyByInverseDiagonalMatrix(mat: IndexedRowMatrix, diagonal: Vector): IndexedRowMatrix = {
    val sArr = diagonal.toArray
    new IndexedRowMatrix(mat.rows.map(indexedRow => {
      val vecArr = indexedRow.vector.toArray
      val newArr = vecArr.indices.toArray.map(i => vecArr(i) / sArr(i))
      IndexedRow(indexedRow.index, Vectors.dense(newArr))
    }))
  }

}
