package scripts.sessionsNtests.slick

import slick.jdbc.PostgresProfile.api._

// Definition of the COFFEES table
class Coffees(tag: Tag) extends Table[(String, Int, Double, Int, Int)](tag, Some("predictions"), "COFFEES") {
  def name = column[String]("COF_NAME", O.PrimaryKey)
  def supID = column[Int]("SUP_ID")
  def price = column[Double]("PRICE")
  def sales = column[Int]("SALES")
  def total = column[Int]("TOTAL")
  def * = (name, supID, price, sales, total)
  // A reified foreign key relation that can be navigated to create a join
  val suppliers = TableQuery[Suppliers]
  def supplier = foreignKey("SUP_FK", supID, suppliers)(_.id)
}
