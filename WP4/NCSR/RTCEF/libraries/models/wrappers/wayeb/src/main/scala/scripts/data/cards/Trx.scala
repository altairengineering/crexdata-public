package scripts.data.cards

object Trx {
  def apply(
             timestamp: Long,
             amount: Double,
             pan: String,
             isFraud: Int,
             fraudType: Int
           ): Trx =
    new Trx(
      timestamp,
      amount,
      pan,
      isFraud,
      fraudType
    )
}

class Trx(
           val timestamp: Long,
           val amount: Double,
           val pan: String,
           val isFraud: Int,
           val fraudType: Int
         ) {

}
