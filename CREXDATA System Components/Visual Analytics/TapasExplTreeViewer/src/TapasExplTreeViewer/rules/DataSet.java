package TapasExplTreeViewer.rules;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class DataSet {
  public String versionLabel="A";
  public String filePath=null, description=null;
  public String fieldNames[]=null;
  public int idIdx=-1, nameIdx=-1, numIdx=-1, classLabelIdx=-1, classNumIdx=-1;

  public ArrayList<DataRecord> records=null;
  /**
   * To compare predictions made by different versions of a model,
   * a new version of the DataSet is created before applying a model.
   * The field previousVersion keeps a reference to the previous version
   * of this data set. The list childVersions includes all versions
   * derived from the given version of the DataSet.
   */

  public DataSet previousVersion=null;
  public ArrayList<DataSet> childVersions=null;

  /**
   * Creates a new instance of DataSet with a reference to this instance.
   * The new instance contains new versions of the data records.
   * @return a new instance of DataSet with a reference to this instance.
   */

  public DataSet makeNewVersion() {
    DataSet ds=new DataSet();
    copyFields(ds);
    if (records!=null && !records.isEmpty()) {
      ds.records = new ArrayList<DataRecord>(records.size());
      for (int i=0; i<records.size(); i++)
        ds.records.add(records.get(i).makeNewVersion());
    }
    if (childVersions==null)
      childVersions=new ArrayList<DataSet>(10);
    childVersions.add(ds);
    ds.versionLabel=versionLabel+"."+childVersions.size();
    return ds;
  }

  public void copyFields(DataSet ds) {
    if (ds==null)
      return;
    ds.previousVersion=this;
    ds.filePath=filePath;
    ds.fieldNames=fieldNames;
    ds.idIdx=idIdx; ds.nameIdx=nameIdx; ds.numIdx=numIdx;
    ds.classLabelIdx=classLabelIdx; ds.classNumIdx=classNumIdx;
  }
  
  /**
   * Puts all vesrions from the descendants hierarchy into a linear list
   */
  public static void putAllVersionsInList(DataSet ds, ArrayList<DataSet> list) {
    if (list==null || ds==null)
      return;
    if (!list.contains(ds))
      list.add(ds);
    if (ds.childVersions==null || ds.childVersions.isEmpty())
      return;
    for (DataSet child:ds.childVersions)
      putAllVersionsInList(child,list);
  }

  /**
   * Finds the original dataset at the beginning of the version chain
   * @return
   */
  public DataSet getOriginalVersion() {
    DataSet ds=this;
    while (ds.previousVersion!=null)
      ds=ds.previousVersion;
    return ds;
  }

  /**
   * Determines the type of the target variable, i.e., class (category) or real value
   */
  public byte determineTargetType() {
    if (records == null || records.isEmpty()) return DataRecord.NO_TARGET;
    for (DataRecord data:records) {
      byte type=data.getTargetType();
      if (type>DataRecord.NO_TARGET)
        return type;
    }
    return DataRecord.NO_TARGET;
  }

  public double[] getTargetMinMax() {
    if (records == null || records.isEmpty())
      return null;
    double minmax[]=null;
    for (DataRecord data:records) {
      if (data.origClassIdx >=0 || !Double.isNaN(data.origValue))  {
        double v=(!Double.isNaN(data.origValue))?data.origValue :data.origClassIdx;
        if (minmax==null) {
          minmax=new double[2];
          minmax[0]=minmax[1]=v;
        }
        else
          if (v<minmax[0]) minmax[0]=v; else
          if (v>minmax[1]) minmax[1]=v;
      }
    }
    return minmax;
  }

  public double[] getPredictionMinMax() {
    if (records == null || records.isEmpty())
      return null;
    double minmax[]=null;
    for (DataRecord data:records) {
      byte type=data.getPredictionType();
      if (type==DataRecord.NO_TARGET)
        continue;
      double v=(type==DataRecord.CLASS_TARGET)?data.predictedClassIdx:
          (type==DataRecord.VALUE_TARGET)?data.predictedValue:
              (type==DataRecord.RANGE_TARGET && data.predictedValueRange!=null)?data.predictedValueRange[0]:Double.NaN;
      if (Double.isNaN(v))
        continue;
      double v2=(type==DataRecord.RANGE_TARGET)?data.predictedValueRange[1]:v;
      if (minmax==null) {
        minmax=new double[2];
        minmax[0]=v; minmax[1]=v2;
      }
      else
      if (v<minmax[0]) minmax[0]=v; else
      if (v2>minmax[1]) minmax[1]=v2;
    }
    return minmax;
  }

  public Integer[] getPredictionKeys() {
    ArrayList<Integer> keys=new ArrayList<Integer>(20);
    for (DataRecord r:records)
      if (r.predictions!=null)
        for (Integer key:r.predictions.keySet())
          if (!keys.contains(key))
            keys.add(key);
    if (keys.isEmpty())
      return null;
    Collections.sort(keys);
    return keys.toArray(new Integer[keys.size()]);
  }

  public int getMaxPredictionWeight(int key) {
    int max=0;
    for (DataRecord r:records)
      if (r.predictions!=null) {
        Integer count=r.predictions.get(key);
        if (count!=null && count > max)
          max = count;
      }
    return max;
  }
  /**
   * Finds the type of model predictions made for the data records
   */
  public byte determinePredictionType() {
    if (records == null || records.isEmpty()) return DataRecord.NO_TARGET;
    for (DataRecord data:records)
      if (data.getPredictionType()>DataRecord.NO_TARGET)
        return data.getPredictionType();
    return DataRecord.NO_TARGET;
  }

  public byte determineFeatureType(String featureName) {
    if (records==null || records.isEmpty())
      return -1;
    for (DataRecord data:records) {
      DataElement item=data.getDataElement(featureName);
      if (item!=null && item.hasAnyValue())
        return item.getDataType();
    }
    return -1;
  }

  public double[] findMinMax(String featureName) {
    if (records == null || records.isEmpty())
      return null;
    double minmax[] = null;
    for (DataRecord data : records) {
      DataElement item = data.getDataElement(featureName);
      if (item != null && item.hasAnyValue()) {
        byte type = item.getDataType();
        if (type != DataElement.INTEGER && type != DataElement.REAL)
          return null;
        double v = item.getDoubleValue();
        if (!Double.isNaN(v)) {
          if (minmax==null) {
            minmax=new double[2];
            minmax[0]=minmax[1]=v;
          }
          else
            if (v<minmax[0]) minmax[0]=v; else
              if (v>minmax[1]) minmax[1]=v;
        }
      }
    }
    return minmax;
  }

  public DataSet extractWronglyPredicted(){
    if (determineTargetType()==DataRecord.NO_TARGET)
      return null; //no target to predict
    if (determinePredictionType()==DataRecord.NO_TARGET)
      return null; //no predictions available
    ArrayList<DataRecord> wrong=null;
    for (DataRecord data:records)
      if (!data.hasCorrectPrediction()) {
        if (wrong==null)
          wrong=new ArrayList<DataRecord>(records.size()/3);
        wrong.add(data);
      }
    if (wrong==null)
      return null;
    DataSet ds=new DataSet();
    ds.records=wrong;
    ds.fieldNames=fieldNames;
    ds.idIdx=idIdx; ds.nameIdx=nameIdx; ds.numIdx=numIdx;
    ds.classLabelIdx=classLabelIdx; ds.classNumIdx=classNumIdx;
    return ds;
  }

  /**
   * Exports the data records to a CSV file, including prediction results based on the prediction type.
   */
  public boolean exportToCSV(String outputFilePath) {
    if (fieldNames==null || records==null || records.isEmpty()) {
      System.out.println("No data in the dataset!");
      return false;
    }
    try (FileWriter writer = new FileWriter(outputFilePath)) {
      // Determine the prediction type for the data set
      byte predictionType = determinePredictionType();

      // Create a list of header fields, including predicted fields based on the prediction type
      ArrayList<String> headerFields = new ArrayList<String>(fieldNames.length+2);
      if (fieldNames != null) {
        for (String field : fieldNames) {
          headerFields.add(field);
        }
      }

      // Add additional fields based on the prediction type
      switch (predictionType) {
        case DataRecord.CLASS_TARGET:
          headerFields.add("PredictedClass");
          break;
        case DataRecord.VALUE_TARGET:
          headerFields.add("PredictedValue");
          break;
        case DataRecord.RANGE_TARGET:
          headerFields.add("PredictedValueLowerBound");
          headerFields.add("PredictedValueUpperBound");
          break;
        default:
          break; // No additional fields needed for NO_TARGET
      }

      // Write the header row to the CSV file
      writer.append(String.join(",", headerFields));
      writer.append("\n");

      // Write each data record to the CSV file
      for (DataRecord record : records) {
        ArrayList<String> rowValues = new ArrayList<String>(headerFields.size());
        for (int i=0; i<fieldNames.length; i++)
          if (i==idIdx) rowValues.add(record.id); else
          if (i==nameIdx) rowValues.add(record.name); else
          if (i==numIdx) rowValues.add(Integer.toString(record.idx)); else
          if (i==classLabelIdx) rowValues.add(record.trueClassLabel); else
          if (i==classNumIdx) rowValues.add(Integer.toString(record.origClassIdx));
          else {
            DataElement item=record.getDataElement(fieldNames[i]);
            if (item==null)
              rowValues.add("");
            else {
              String sValue=item.getStringValue();
              rowValues.add((sValue==null)?"":sValue);
            }
          }
        switch (predictionType) {
          case DataRecord.CLASS_TARGET:
            rowValues.add((record.predictedClassIdx>=0)?Integer.toString(record.predictedClassIdx):"");
            break;
          case DataRecord.VALUE_TARGET:
            rowValues.add((Double.isNaN(record.predictedValue))?"":String.format("%.6f",record.predictedValue));
            break;
          case DataRecord.RANGE_TARGET:
            if (record.predictedValueRange==null) {
              rowValues.add(""); rowValues.add("");
            }
            else {
              rowValues.add(String.format("%.6f",record.predictedValueRange[0]));
              rowValues.add(String.format("%.6f",record.predictedValueRange[1]));
            }
            break;
        }

        // Write the row to the CSV file
        writer.append(String.join(",", rowValues));
        writer.append("\n");
      }

      writer.flush();
      return true; // Successful export
    } catch (IOException e) {
      e.printStackTrace();
      return false; // Failed export
    }
  }
}
