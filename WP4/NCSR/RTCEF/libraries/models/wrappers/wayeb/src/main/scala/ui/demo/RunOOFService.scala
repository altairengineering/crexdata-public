package ui.demo

import py4j.GatewayServer
import ui.WayebConfig
import utils.WrappedMethods4Python

import java.net.InetAddress
import scala.collection.mutable.ListBuffer

object RunOOFService {

  def main(config: WayebConfig): Unit = {
    val java_port = config.javaPort
    val python_port = config.pythonPort

    val gatewayServer = new GatewayServer.GatewayServerBuilder(new WrappedMethods4Python()).
      javaPort(java_port).
      javaAddress(InetAddress.getByName("127.0.0.1")).
      callbackClient(python_port, InetAddress.getByName("127.0.0.1")).build()
    gatewayServer.start()
    println("Wayeb OOF service server side started.")
  }
}
