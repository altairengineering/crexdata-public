package TapasExplTreeViewer.ui;

import TapasDataReader.CountMatrix;

import javax.swing.table.AbstractTableModel;

public class TableOfIntegersModel extends AbstractTableModel {
  public String colNames[]=null;
  public String rowNames[]=null;
  public Integer cellValues[][]=null;
  
  public TableOfIntegersModel(String colNames[], String rowNames[], Integer cellValues[][]) {
    this.colNames=colNames; this.rowNames=rowNames; this.cellValues=cellValues;
  }
  
  public TableOfIntegersModel(CountMatrix matrix) {
    if (matrix==null)
      return;
    this.colNames=matrix.colNames; this.rowNames=matrix.rowNames; this.cellValues=matrix.cellValues;
  }
  
  public int getColumnCount() {
    if (colNames==null)
      return 0;
    return colNames.length;
  }
  
  public int getRowCount() {
    if (rowNames==null)
      if (cellValues!=null)
        return cellValues.length;
      else
        return 0;
    return rowNames.length;
  }
  
  public String getColumnName(int col) {
    if (colNames!=null)
      return colNames[col];
    return null;
  }
  
  public Class getColumnClass(int c) {
    return getValueAt(0, c).getClass();
  }
  
  public Object getValueAt(int row, int col) {
    if (cellValues==null)
      return (rowNames!=null && col==0)?rowNames[row]:null;
    if (rowNames!=null)
      return (col == 0)?rowNames[row]:cellValues[row][col-1];
    return cellValues[row][col];
  }
  
  /**
 
   class TwoColumnsTableModel extends AbstractTableModel {
     String columns[] = null;
     String colStr[] = null;
     Integer colInt[] = null;
   
     public TwoColumnsTableModel(String columns[], String colStr[], Integer colInt[]) {
       this.columns = columns;
       this.colStr = colStr;
       this.colInt = colInt;
     }
   
     public int getColumnCount() {
       return columns.length;
     }
   
     public int getRowCount() {
       return colStr.length;
     }
   
     public String getColumnName(int col) {
       return columns[col];
     }
   
     public Class getColumnClass(int c) {
       return getValueAt(0, c).getClass();
     }
   
     public Object getValueAt(int row, int col) {
       if (col == 0)
       return colStr[row];
       else
       return colInt[row];
     }
   }
  
   */
}
