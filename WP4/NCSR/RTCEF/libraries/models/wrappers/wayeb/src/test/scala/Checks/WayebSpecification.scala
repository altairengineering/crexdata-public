package Checks

import org.scalacheck.Properties

/**
  * Created by elias on 6/30/17.
  */
object WayebSpecification extends Properties("Wayeb") {
  include(DisSpec)
}
