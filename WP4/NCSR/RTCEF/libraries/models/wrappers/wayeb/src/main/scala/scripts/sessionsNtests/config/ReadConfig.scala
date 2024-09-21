package scripts.sessionsNtests.config

import com.typesafe.config.{Config, ConfigFactory}

object ReadConfig {
  def main(args: Array[String]): Unit = {
    val config: Config = ConfigFactory.load()

    val defaultHorizon = config.getInt("default.horizon")

    println(defaultHorizon)
  }

}
