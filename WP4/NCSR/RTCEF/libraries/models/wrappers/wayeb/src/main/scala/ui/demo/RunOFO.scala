package ui.demo
import py4j.GatewayServer
import utils.WrappedMethods4Python

import scala.collection.mutable.ListBuffer
import java.net.InetAddress

object RunOFO {
  def main(): Unit = {

//    val java_ports = Array(25334, 25336, 25338, 25340, 25342, 25344, 25346, 25348)
//    val python_ports = Array(25333, 25335, 25337, 25339, 25341, 25343, 25345, 25347)
    val java_ports = Array(25334)
    val python_ports = Array(25333)
    val n_servers = java_ports.length

    val gatewayServers = new ListBuffer[GatewayServer]
    for(index <- 0 until n_servers){
      val gatewayServerI = new GatewayServer.GatewayServerBuilder(new WrappedMethods4Python()).javaPort(java_ports(index)).javaAddress(InetAddress.getByName("127.0.0.1")).callbackClient(python_ports(index), InetAddress.getByName("127.0.0.1")).build()
      gatewayServerI.start()
      gatewayServers += gatewayServerI
    }
    println("Gateway Servers Started")
  }
}
