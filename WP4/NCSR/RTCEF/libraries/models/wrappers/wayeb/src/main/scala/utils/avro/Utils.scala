package utils.avro

import org.apache.avro.Schema

import java.io.{File, FileNotFoundException}
import java.util.Scanner

object Utils {
  def parseSchemaFromPath(schemaPath:String):Schema = {

    var schemaString: String = null
    try schemaString = new Scanner(new File(schemaPath)).useDelimiter("\\Z").next
    catch {
      case e: FileNotFoundException =>
        throw new RuntimeException(e)
    }

    new Schema.Parser().parse(schemaString)
  }
}
