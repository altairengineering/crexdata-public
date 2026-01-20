package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;

import javax.swing.table.AbstractTableModel;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Map;

public class DataForRuleTableModel extends AbstractTableModel {

  public CommonExplanation cEx=null;
  public Hashtable<String,float[]> attrMinMax=null;
  public ArrayList<String> listOfFeatures=null;


  public DataForRuleTableModel (CommonExplanation cEx, ArrayList<String> listOfFeatures_parent, Hashtable<String, float[]> attrMinMax) {
    this.cEx=cEx;
    this.attrMinMax=attrMinMax;
    this.listOfFeatures=new ArrayList<String>(listOfFeatures_parent.size());
    for (String aName:listOfFeatures_parent) {
      int idx=-1;
      for (int j = 0; idx==-1 && j < cEx.eItems.length; j++)
        if (aName.equals(cEx.eItems[j].attr))
          idx=j;
      if (idx!=-1)
        listOfFeatures.add(aName);
    }
  }

  public String columnNames[] = {"Id","ok?","Action"};

  public int getColumnCount() {
    return columnNames.length+listOfFeatures.size();
  }
  public String getColumnName(int col) {
    return ((col<columnNames.length) ? columnNames[col] :
            listOfFeatures.get(col-columnNames.length));
  }
  public int getRowCount() {
    return cEx.getApplicationsCount(); //exList.size();
  }

  public Class getColumnClass(int c) {
    return (getValueAt(0, c) == null) ? null : getValueAt(0, c).getClass();
  }

  public Object getValueAt(int row, int col) {
    if (cEx.getApplicationsCount()<1)
      return 0;
    switch (col) {
      case 0:
        return cEx.applications[row].data.FlightID;
      case 1:
        return cEx.action==cEx.applications[row].data.action;
      case 2:
        return cEx.applications[row].data.action;
      default:
        int idx=col-columnNames.length;
        String aName=listOfFeatures.get(idx);
        double values[]={Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN};
        for (int i=0; i<cEx.eItems.length; i++)
          if (aName.equals(cEx.eItems[i].attr)) {
            if (Double.isNaN(values[0]) || values[0]>cEx.eItems[i].interval[0])
              values[0]=cEx.eItems[i].interval[0];
            if (Double.isNaN(values[1]) || values[1]<cEx.eItems[i].interval[1])
              values[1]=cEx.eItems[i].interval[1];
          }
        if (attrMinMax !=null && !Double.isNaN(values[0]) || !Double.isNaN(values[1]))  {
          float minmax[]= attrMinMax.get(aName);
          if (minmax!=null) {
            values[2]=minmax[0];
            values[3]=minmax[1];
            if (Double.isNaN(values[0]) || Double.isInfinite(values[0]))
              values[0]=values[2];
            if (Double.isNaN(values[1]) || Double.isInfinite(values[1]))
              values[1]=values[3];
          }
        }
        for (int i=0; i<cEx.applications[row].data.eItems.length; i++)
          if (aName.equals(cEx.applications[row].data.eItems[i].attr)) {
            values[4]=cEx.applications[row].data.eItems[i].value;
            return values;
          }
    }
    return 0;
  }

}
