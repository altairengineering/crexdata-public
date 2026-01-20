package TapasExplTreeViewer.ui;

import TapasExplTreeViewer.rules.DataElement;
import TapasExplTreeViewer.rules.DataRecord;
import TapasExplTreeViewer.rules.DataSet;

import javax.swing.table.AbstractTableModel;
import java.util.ArrayList;

public class DataTableModel extends AbstractTableModel {
  public int nStandardColumns=4, nClasses=0;
  public Integer classes[]=null;
  public DataSet dataSet=null;
  public String featureNames[]=null;
  protected String[] columnNames=null;
  public ArrayList<DataRecord> records=null;

  public DataTableModel(DataSet dataSet, String featureNames[]) {
    this.dataSet=dataSet;
    if (dataSet==null || dataSet.records==null || dataSet.records.isEmpty())
      return;
    this.records = dataSet.records;
    this.featureNames = (featureNames==null)?dataSet.fieldNames:featureNames;

    if (dataSet.determinePredictionType()==DataRecord.CLASS_TARGET) {
      classes=dataSet.getPredictionKeys();
      if (classes!=null && classes.length>1)
        nClasses=classes.length;
    }

    // Define column names: Record ID, True Class, Predicted Class/Value, and Features
    columnNames = new String[nStandardColumns +nClasses + featureNames.length];

    columnNames[0] = "Record ID";
    columnNames[1] = "Original Class/Value";
    columnNames[2] = "Predicted Class/Value";
    columnNames[3] = "Match?";
    if (nClasses>0) {
      for (int i = 0; i < nClasses; i++)
        columnNames[nStandardColumns + i] = "Weight " + classes[i];
      nStandardColumns += nClasses;
    }

    System.arraycopy(featureNames, 0, columnNames, nStandardColumns, featureNames.length);
  }

  @Override
  public int getRowCount() {
    return records.size();
  }

  @Override
  public int getColumnCount() {
    return columnNames.length;
  }

  @Override
  public String getColumnName(int columnIndex) {
    return columnNames[columnIndex];
  }

  @Override
  public Object getValueAt(int rowIndex, int columnIndex) {
    DataRecord record = records.get(rowIndex);

    if (columnIndex == 0) {
      return record.id;
    } else if (columnIndex == 1) {
      return (record.origClassIdx >=0) ? record.origClassIdx : record.origValue;
    } else if (columnIndex == 2) {
      switch (record.getPredictionType()) {
        case DataRecord.CLASS_TARGET:
          return record.predictedClassIdx;
        case DataRecord.VALUE_TARGET:
          return record.predictedValue;
        case DataRecord.RANGE_TARGET:
          return "[" + record.predictedValueRange[0] + " - " + record.predictedValueRange[1] + "]";
        default:
          return "";
      }
    } else if (columnIndex == 3) {
      return (checkPrediction(record)) ? "yes" : "no";
    } else if (columnIndex<nStandardColumns && nClasses>0)  {
      if (record.predictions==null || record.predictions.isEmpty())
        return 0;
      Integer count=record.predictions.get(classes[columnIndex-nStandardColumns+nClasses]);
      if (count==null)
        return 0;
      return count;
    } else if (columnIndex>=nStandardColumns){
      // Handle feature columns
      int featureIndex = columnIndex - nStandardColumns;
      String featureName = featureNames[featureIndex];
      DataElement dataElement = record.getDataElement(featureName);

      if (dataElement != null) {
        switch (dataElement.dataType) {
          case DataElement.CATEGORY:
            return dataElement.stringValue;
          case DataElement.INTEGER:
            return dataElement.intValue > Integer.MIN_VALUE ? dataElement.intValue : "";
          case DataElement.REAL:
            return !Double.isNaN(dataElement.doubleValue) ? dataElement.doubleValue : "";
          default:
            return "";
        }
      } else {
        return ""; // If no value, return empty string
      }
    }
    return "";
  }
  
  public boolean checkPrediction(DataRecord r) {
    if (r.origClassIdx>=0)
      return r.origClassIdx==r.predictedClassIdx;
    if (!Double.isNaN(r.origValue)) {
      if (!Double.isNaN(r.predictedValue))
        return r.predictedValue==r.origValue;
      if (r.predictedValueRange!=null)
        return r.origValue>=r.predictedValueRange[0] &&
            r.origValue<=r.predictedValueRange[1];
      return false;
    }
    if (r.origValueRange!=null) {
      if (!Double.isNaN(r.predictedValue))
        return r.predictedValue>=r.origValueRange[0] &&
            r.predictedValue<=r.origValueRange[1];
      if (r.predictedValueRange!=null)
        return r.predictedValue==r.origValueRange[0] &&
            r.predictedValue==r.origValueRange[1];
    }
    return false;
  }

  public boolean isNumericColumn(int columnIndex) {
    if (columnIndex==0 || columnIndex==3)
      return false;
    if (columnIndex==1)
      return true;
    if (columnIndex==2)
      return dataSet.determinePredictionType()!=DataRecord.RANGE_TARGET;
    if (columnIndex<nStandardColumns && nClasses>0)
      return true;
    if (columnIndex>=nStandardColumns) {
      int featureIndex = columnIndex - nStandardColumns;
      byte type = dataSet.determineFeatureType(featureNames[featureIndex]);
      return type == DataElement.INTEGER || type == DataElement.REAL;
    }
    return false;
  }

  public Class<?> getColumnClass(int columnIndex) {
    if (columnIndex==0 || columnIndex==3)
      return String.class;
    if (columnIndex==1)
      return Number.class;
    if (columnIndex==2)
      return  (dataSet.determinePredictionType()!=DataRecord.RANGE_TARGET)?Number.class:String.class;
    if (columnIndex<nStandardColumns && nClasses>0)
      return Integer.class;
    if (columnIndex>=nStandardColumns) {
      int featureIndex = columnIndex - nStandardColumns;
      byte type = dataSet.determineFeatureType(featureNames[featureIndex]);
      if (type == DataElement.INTEGER)
        return Integer.class;
      if (type == DataElement.REAL)
        return Number.class;
    }
    return String.class;
  }

  public double[] getColumnMinMax(int columnIndex) {
    if (columnIndex==0 || columnIndex==3)
      return null;
    if (columnIndex==1)
      return dataSet.getTargetMinMax();
    if (columnIndex==2)
      return dataSet.getPredictionMinMax();
    if (columnIndex<nStandardColumns && nClasses>0) {
      double mm[]={0, dataSet.getMaxPredictionWeight(classes[columnIndex-nStandardColumns+nClasses])};
      return mm;
    }
    if (columnIndex>=nStandardColumns) {
      int featureIndex = columnIndex - nStandardColumns;
      return dataSet.findMinMax(featureNames[featureIndex]);
    }
    return null;
  }

  @Override
  public boolean isCellEditable(int rowIndex, int columnIndex) {
    return false; // All cells are non-editable for this viewer
  }
}
