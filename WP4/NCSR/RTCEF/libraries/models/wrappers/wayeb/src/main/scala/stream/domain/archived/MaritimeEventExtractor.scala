/*package stream.domain.maritime.datacron.maritime

import data.{InstantaneousEvent, InstantaneousEventExtractor}

class MaritimeEventExtractor extends InstantaneousEventExtractor {
  override def apply(entry: String): Option[InstantaneousEvent] = {
    val x = entry.split(',')
    Some(InstantaneousEvent("MaritimeMessage", x.dropRight(1), x(x.length - 1).toLong))
  }
}*/
