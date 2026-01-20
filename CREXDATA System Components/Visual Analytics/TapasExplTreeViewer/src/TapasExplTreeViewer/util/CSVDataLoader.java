package TapasExplTreeViewer.util;

import TapasExplTreeViewer.SeeRules;
import TapasExplTreeViewer.rules.DataElement;
import TapasExplTreeViewer.rules.DataRecord;
import TapasExplTreeViewer.rules.DataSet;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class CSVDataLoader {

  /**
   * Reads a CSV file and converts each row into a DataRecord object.
   *
   * @param filePath The path to the CSV file.
   * @return An ArrayList of DataRecord objects.
   * @throws IOException If there is an error reading the file.
   */
  public static DataSet loadDataFromCSVFile(String filePath) throws IOException {
    if (filePath==null)
      return null;
    ArrayList<DataRecord> dataRecords = new ArrayList<DataRecord>(10000);
    String[] headers=null;
    int idIdx=-1, nameIdx=-1, numIdx=-1, classLabelIdx=-1, classNumIdx=-1;

    try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
      String line = reader.readLine();
      if (line == null) {
        throw new IOException("Empty CSV file");
      }

      // Split the header line to get the field names
      headers = line.split(",");

      // Validate that we have at least the identifier and some features
      if (headers.length < 2) {
        throw new IOException("The CSV file must have at least an identifier and one feature.");
      }

      for (int i=0; i<headers.length; i++) {
        String h=headers[i].toUpperCase();
        if (h.equals("N") || h.equals("NUM"))
          numIdx=i;
        else
        if (h.equals("ID") ||
            h.equals("IDENT") ||
            h.equals("IDENTIFIER"))
          idIdx=i;
        else
        if (h.equals("NAME"))
          nameIdx=i;
        else
        if (h.contains("CLASS") && !h.contains("PREDICT"))
          if (h.endsWith("N") || h.endsWith("NUM")  || h.endsWith("NUMBER") || h.endsWith("INDEX"))
            classNumIdx=i;
          else
            classLabelIdx=i;
      }

      // Read each row and convert it to a DataRecord
      int rowIndex = 0;
      while ((line = reader.readLine()) != null) {
        String[] values = line.split(",");

        // Ensure the row has at least as many values as headers
        if (values.length < 2) {
          System.err.println("Skipping incomplete row at index " + rowIndex);
          continue;
        }

        // Create a new DataRecord object using the identifier and name (if present)
        String recordId = (idIdx>=0)?values[idIdx]:Integer.toString(rowIndex+1);
        String recordName = (nameIdx>=0) ? values[nameIdx] : null;
        int recordIdx=rowIndex;
        if (numIdx>=0)
          try {
            recordIdx=Integer.parseInt(values[numIdx]);
          } catch (Exception ex) {}
        DataRecord record = new DataRecord(recordIdx, recordId, recordName);
        if (classLabelIdx>=0) {
          record.trueClassLabel = values[classLabelIdx];
          try {
            record.origClassIdx =Integer.parseInt(record.trueClassLabel);
          } catch (Exception ex) {}
        }
        if (classNumIdx>=0)
          try {
            record.origClassIdx =Integer.parseInt(values[classNumIdx]);
          } catch (Exception ex) {}

        // Add the feature data to the DataRecord
        for (int i = 0; i < headers.length && i<values.length; i++) {
          if (i==idIdx || i==nameIdx || i==numIdx || i==classLabelIdx || i==classNumIdx)
            continue;
          byte dataType = inferDataType(values[i]);
          record.addDataElement(headers[i], values[i], dataType);
        }

        dataRecords.add(record);
        rowIndex++;
      }
    }
    if (dataRecords.isEmpty())
      return null;
    DataSet ds=new DataSet();
    ds.filePath=filePath;
    ds.fieldNames=headers;
    ds.records=dataRecords;
    ds.idIdx=idIdx; ds.nameIdx=nameIdx; ds.numIdx=numIdx;
    ds.classLabelIdx=classLabelIdx; ds.classNumIdx=classNumIdx;
    return ds;
  }

  /**
   * Infers the data type of a value (CATEGORY, INTEGER, or REAL).
   *
   * @param value The value to infer the data type for.
   * @return The inferred data type as a byte.
   */
  public static byte inferDataType(String value) {
    if (value == null || value.isEmpty()) {
      return DataElement.CATEGORY;
    }
    try {
      Integer.parseInt(value);
      return DataElement.INTEGER;
    } catch (NumberFormatException e1) {
      try {
        Double.parseDouble(value);
        return DataElement.REAL;
      } catch (NumberFormatException e2) {
        return DataElement.CATEGORY;
      }
    }
  }

  public static String selectFilePathThroughDialog(boolean openExisting) {
    // Select file with rules
    JFileChooser fileChooser = new JFileChooser();
    fileChooser.setDialogTitle("Specify a file with data to test the rules");
    if (SeeRules.lastUsedDirectory!=null)
      fileChooser.setCurrentDirectory(new File(SeeRules.lastUsedDirectory));
    else {
      String workingDirectory = System.getProperty("user.dir");
      if (workingDirectory != null)
        fileChooser.setCurrentDirectory(new File(workingDirectory));
    }
    // Set the file filter to only show CSV files
    FileNameExtensionFilter filter = new FileNameExtensionFilter("CSV Files", "csv");
    int userSelection = (openExisting)?fileChooser.showOpenDialog(null):
        fileChooser.showSaveDialog(null);

    if (userSelection == JFileChooser.APPROVE_OPTION) {
      File file = fileChooser.getSelectedFile();
      SeeRules.lastUsedDirectory = file.getParent();
      return file.getAbsolutePath();
    }
    return null;
  }
}
