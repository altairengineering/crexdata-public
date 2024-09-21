package scripts.sessionsNtests.slick

import java.sql.{Connection, DriverManager}

object JDBCTest {
  def main(args: Array[String]): Unit = {
    val driver = "org.postgresql.Driver"
    val url = "jdbc:postgresql://localhost/wayeb"
    val username = "postgres"
    val password = "postgres"

    var connection: Connection = null

    try {
      // make the connection
      Class.forName(driver)
      connection = DriverManager.getConnection(url, username, password)

      // create the statement, and run the select query
      val statement = connection.createStatement()
      val resultSet = statement.executeQuery("SELECT ts, isCorrect FROM approachingbrestport.predictions")
      while (resultSet.next()) {
        val ts = resultSet.getString("ts")
        val isCorrect = resultSet.getString("isCorrect")
        println("ts, correct = " + ts + ", " + isCorrect)
      }
    } catch {
      case e: Throwable => e.printStackTrace
    }
    connection.close()
  }

  /*
  classOf[org.postgresql.Driver]
    val con_st = "jdbc:postgresql://localhost:5432/wayeb?user=postgres&password=postgres"
    val conn = DriverManager.getConnection(con_st)
    try {
      val stm = conn.createStatement(ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY)

      val rs = stm.executeQuery("SELECT * from approachingbrestport.predictions")

      while(rs.next) {
        println(rs.getString("ts"))
      }
    } finally {
      conn.close()
    }
   */
}
