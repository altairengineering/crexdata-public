package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;
import TapasDataReader.Explanation;
import TapasExplTreeViewer.clustering.ClustersAssignments;
import TapasExplTreeViewer.rules.UnitedRule;
import TapasUtilities.MySammonsProjection;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Map;

public class ExListTableModel extends AbstractTableModel implements ChangeListener {
  /**
   * The explanations to show
   */
  public ArrayList<CommonExplanation> exList=null;
  public Hashtable<String,float[]> attrMinMax =null;
  public ArrayList<String> listOfFeatures=null;
  public int order[]=null, clusters[]=null;
  public String columnNamesUnited[]={"N right covers","N wrong covers","Coherence",
      "N united rules","Depth of hierarchy"};
  public ArrayList<String> listOfColumnNames=null;
  public double qMin=Double.NaN, qMax=Double.NaN;

  boolean drawValuesOrStatsForIntervals=false;
  public void setDrawValuesOrStatsForIntervals (boolean drawValuesOrStatsForIntervals) {
    this.drawValuesOrStatsForIntervals=drawValuesOrStatsForIntervals;
  }
  
  public ExListTableModel(ArrayList<CommonExplanation> exList,
                          Hashtable<String,float[]> attrMinMax,
                          ArrayList<String> orderedFeatureNames) {
    this.exList=exList;
    this.attrMinMax =attrMinMax;
    Hashtable<String,Integer> attrUses=new Hashtable<String,Integer>(50);
    boolean hasUnitedRules=false;
    int minClass=-1, maxClass=-1;
    int maxNUses=0, maxRight=0, maxWrong=0, minTreeId=-1, maxTreeId=-1, minTreeCluster=-1, maxTreeCluster=-1;
    boolean rulesHaveQIntervals=false;
    boolean rulesHaveCategories=false;

    for (int i=0; i<exList.size(); i++) {
      CommonExplanation cEx = exList.get(i);
      hasUnitedRules=hasUnitedRules || (cEx instanceof UnitedRule);
      rulesHaveCategories= rulesHaveCategories || cEx.category!=null;

      for (int j = 0; j < cEx.eItems.length; j++) {
        Integer count=attrUses.get(cEx.eItems[j].attr);
        if (count==null)
          attrUses.put(cEx.eItems[j].attr,1);
        else
          attrUses.put(cEx.eItems[j].attr,count+1);
      }
      boolean qEqualsAction=false;
      boolean hasMinQ=!Float.isNaN(cEx.minQ), hasMaxQ=!Float.isNaN(cEx.maxQ);
      if (cEx.action>=0) {
        if (minClass<0 || minClass>cEx.action)
          minClass=cEx.action;
        if (maxClass<cEx.action)
          maxClass=cEx.action;
        qEqualsAction=hasMinQ && hasMaxQ && cEx.minQ==cEx.action && cEx.maxQ==cEx.action;
      }
      if (hasMinQ && !qEqualsAction)
        if (Double.isNaN(qMin) || qMin>cEx.minQ)
          qMin=cEx.minQ;
      if (hasMaxQ && !qEqualsAction)
        if (Double.isNaN(qMax) || qMax<cEx.maxQ)
          qMax=cEx.maxQ;
      if (hasMinQ && hasMaxQ && Float.isNaN(cEx.meanQ))   {
        if (!Double.isNaN(cEx.sumQ)) {
          int count=(cEx instanceof UnitedRule)?((UnitedRule)cEx).countFromRules():cEx.getUsesCount();
          if (count<1) count=1;
          cEx.meanQ = (float) (cEx.sumQ / count);
        }
        else
          cEx.meanQ=(cEx.minQ+cEx.maxQ)/2;
      }
      rulesHaveQIntervals = rulesHaveQIntervals || (!qEqualsAction && hasMinQ && hasMaxQ && cEx.maxQ>cEx.minQ);

      if (maxNUses<cEx.getUsesCount())
        maxNUses=cEx.getUsesCount();
      if (maxRight<cEx.nCasesRight)
        maxRight=cEx.nCasesRight;
      if (maxWrong<cEx.nCasesWrong)
        maxWrong=cEx.nCasesWrong;

      if (cEx.treeId>=0) {
        if (minTreeId < 0 || minTreeId > cEx.treeId)
          minTreeId = cEx.treeId;
        if (maxTreeId < cEx.treeId)
          maxTreeId = cEx.treeId;
      }
      if (cEx.treeCluster>=0) {
        if (minTreeCluster<0 || minTreeCluster>cEx.treeCluster)
          minTreeCluster=cEx.treeCluster;
        if (maxTreeCluster<cEx.treeCluster)
          maxTreeCluster=cEx.treeCluster;
      }
    }
    boolean toShowUpperId=false;
    if (hasUnitedRules) {
      int lastUpperN=exList.get(0).upperId;
      for (int i=1; i<exList.size() && !toShowUpperId; i++)
        toShowUpperId=lastUpperN!=exList.get(i).upperId;
    }
    
    listOfFeatures=new ArrayList<String>(attrUses.size());
    for (Map.Entry<String,Integer> entry:attrUses.entrySet()) {
      String aName=entry.getKey();
      if (listOfFeatures.isEmpty())
        listOfFeatures.add(aName);
      else
        if (orderedFeatureNames!=null && !orderedFeatureNames.isEmpty()) {
          int ord=orderedFeatureNames.indexOf(aName);
          if (ord<0)
            listOfFeatures.add(aName);
          else {
            int idx=-1;
            for (int i=0; i<listOfFeatures.size() && idx<0; i++)
              if (orderedFeatureNames.indexOf(listOfFeatures.get(i))>ord)
                idx=i;
            if (idx<0)
              listOfFeatures.add(aName);
            else
              listOfFeatures.add(idx,aName);
          }
        }
        else {
          int count=entry.getValue(), idx=-1;
          for (int i=0; i<listOfFeatures.size() && idx<0; i++)
            if (count>attrUses.get(listOfFeatures.get(i)))
              idx=i;
          if (idx<0)
            listOfFeatures.add(aName);
          else
            listOfFeatures.add(idx,aName);
        }
    }
    listOfColumnNames=new ArrayList<String>(25+listOfFeatures.size());
    listOfColumnNames.add("Id");
    if (rulesHaveCategories)
      listOfColumnNames.add("Category");
    if (maxTreeId>=0)
      listOfColumnNames.add("Tree Id");
    if (maxTreeCluster>=0)
      listOfColumnNames.add("Tree cluster");
    listOfColumnNames.add("Weight");
    if (maxClass>=0)
      listOfColumnNames.add("Class");
    if (!Double.isNaN(qMin) && !Double.isNaN(qMax) && qMax>qMin) {
      listOfColumnNames.add((rulesHaveQIntervals)?"mean value":"value");
      if (rulesHaveQIntervals) {
        listOfColumnNames.add("min");
        listOfColumnNames.add("max");
        listOfColumnNames.add("min_max");
      }
    }
    if (maxNUses>1)
      listOfColumnNames.add("N uses");
    //if (maxRight>0) {
      listOfColumnNames.add("N+");
      listOfColumnNames.add("N-");
      listOfColumnNames.add("+/-");
    //}
    listOfColumnNames.add("order");
    listOfColumnNames.add("cluster");
    listOfColumnNames.add("N conditions");
    listOfColumnNames.add("Rule");

    if (hasUnitedRules) {
      if (toShowUpperId)
        listOfColumnNames.add(1,"upper Id");
      for (int i=0; i<columnNamesUnited.length; i++)
        listOfColumnNames.add(columnNamesUnited[i]);
    }
  }
  public int getColumnCount() {
    return listOfColumnNames.size() + listOfFeatures.size();
  }
  public String getColumnName(int col) {
    return ((col<listOfColumnNames.size()) ? listOfColumnNames.get(col) :
                listOfFeatures.get(col-listOfColumnNames.size()));
  }
  public int getRowCount() {
    return exList.size();
  }

  public int getRuleColumnIdx() {
    int idx=listOfColumnNames.indexOf("Rule");
    if (idx<0)
      idx=listOfColumnNames.indexOf("rule");
    if (idx<0)
      for (int i=0; i<listOfColumnNames.size() && idx<0; i++) {
        String cName=getColumnName(i).toLowerCase();
        if (cName.startsWith("rule") && !cName.contains("id"))
          idx=i;
      }
    return idx;
  }

  public boolean isClusterColumn(int i) {
    if (i>=listOfColumnNames.size())
      return false;
    return getColumnName(i).toLowerCase().equals("cluster");
  }

  public boolean isResultClassColumn(int i) {
    if (i>=listOfColumnNames.size())
      return false;
    String cName=listOfColumnNames.get(i).toLowerCase();
    return cName.equals("class") || cName.equals("action");
  }

  public boolean isMinMaxColumn(int i) {
    if (i>=listOfColumnNames.size())
      return false;
    String cName=listOfColumnNames.get(i).toLowerCase();
    return cName.contains("min") && cName.contains("max");
  }

  public boolean isResultValueColumn(int i) {
    if (i>=listOfColumnNames.size())
      return false;
    String cName=listOfColumnNames.get(i).toLowerCase();
    if (cName.equals("q"))
      return true;
    return cName.contains("value") || cName.contains("mean")  || cName.contains("min") || cName.contains("max");
  }

  public Class getColumnClass(int c) {
    String intColNames[]={"upper id","tree id","tree cluster","class","action",
        "weight","uses","n+","n +","n-","n -","order","cluster","conditions"};
    String floatColNames[]={"value","q","min","max","mean","x","x1d"};
    String catColNames[]={"category","type"};

    String colName=getColumnName(c).toLowerCase();
    for (String name:catColNames)
      if (name.equals(colName))
        return String.class;
    for (String name:intColNames)
      if (name.equals(colName))
        return Integer.class;
    for (String name:floatColNames)
      if (name.equals(colName))
        return Float.class;

    if (colName.contains("mean") || colName.contains("+/-"))
      return Float.class;

    return (getValueAt(0, c) == null) ? null : getValueAt(0, c).getClass();
  }
  
  public void setCusterAssignments(ClustersAssignments clAss) {
    if (clAss==null || clAss.objIndexes==null)
      return;
    if (order==null || order.length!=exList.size())
      order = new int[exList.size()];
    if (clusters==null || clusters.length!=exList.size())
      clusters = new int[exList.size()];
    for (int i=0; i<order.length; i++) {
      order[i] = i; clusters[i]=-1;
    }
    for (int i=0; i<clAss.objIndexes.length; i++) {
      int idx=clAss.objIndexes[i];
      if (idx>=0 && idx<exList.size()) {
        if (order[idx]!=idx) {
          System.out.println("Repeated order assignment to item #"+idx+
                                ": previous value = "+order[idx]+", new value = "+i);
        }
        order[idx]=i;
        clusters[idx]=clAss.clusters[i];
      }
      else
        System.out.println("Trying to assign the ordinal value "+i+" to item #"+idx+
                               " while there are "+exList.size()+" items");
    }
    fireTableDataChanged();
  }
  
  public Object getValueAt(int row, int col) {
    if (row>=exList.size())
      return null;
    CommonExplanation cEx=exList.get(row);

    if (col>=listOfColumnNames.size()) {
      //return the feature value
      int fIdx=col-listOfColumnNames.size();
      String attrName=listOfFeatures.get(fIdx);
      double values[]={Double.NaN,Double.NaN,Double.NaN,Double.NaN};
      for (int i=0; i<cEx.eItems.length; i++)
        if (attrName.equals(cEx.eItems[i].attr)) {
          if (Double.isNaN(values[0]) || values[0]>cEx.eItems[i].interval[0])
            values[0]=cEx.eItems[i].interval[0];
          if (Double.isNaN(values[1]) || values[1]<cEx.eItems[i].interval[1])
            values[1]=cEx.eItems[i].interval[1];
        }
      if (attrMinMax !=null && !Double.isNaN(values[0]) || !Double.isNaN(values[1]))  {
        float minmax[]= attrMinMax.get(attrName);
        if (minmax!=null) {
          values[2]=minmax[0];
          values[3]=minmax[1];
          if (Double.isNaN(values[0]) || Double.isInfinite(values[0]))
            values[0]=values[2];
          if (Double.isNaN(values[1]) || Double.isInfinite(values[1]))
            values[1]=values[3];
        }
      }
      if (!drawValuesOrStatsForIntervals)
        return values;
      int N=0;
      if (cEx.uses!=null)
        for (String s:cEx.uses.keySet()) {
          ArrayList<Explanation> aex = cEx.uses.get(s);
          N += aex.size();
        }
      double v[]=new double[values.length+N];
      for (int j=0; j<values.length; j++)
        v[j]=values[j];
      int j=0;
      if (cEx.uses!=null && !cEx.uses.isEmpty())
        for (String s:cEx.uses.keySet()) {
          ArrayList<Explanation> aex=cEx.uses.get(s);
          for (Explanation ex:aex) {
            int attrIdx=-1;
            for (int i=0; i<ex.eItems.length && attrIdx==-1; i++)
              if (attrName.equals(ex.eItems[i].attr))
                attrIdx=i;
            if (attrIdx!=-1) {
              float vv=ex.eItems[attrIdx].value;
              v[values.length+j]=vv;
            }
            j++;
          }
        }
      return v;
    }
    else {
      if (col==0)
        return new Integer(cEx.numId);
      String colName=getColumnName(col).toLowerCase();
      if (colName.equals("rule"))
        return cEx;
      if (colName.equals("upper id"))
        return new Integer(cEx.upperId);
      if (colName.equals("tree id"))
        return new Integer(cEx.treeId);
      if (colName.equals("tree cluster"))
        return new Integer(cEx.treeCluster);
      if (colName.equals("type") || colName.equals("category"))
        return cEx.category;
      if (colName.equals("class"))
        return new Integer(cEx.action);
      if (colName.equals("weight"))
        return new Integer(cEx.weight);
      if (colName.equals("value") || colName.equals("q") || colName.contains("mean"))
        return new Float(cEx.meanQ);
      if (colName.equals("min"))
        return new Float(cEx.minQ);
      if (colName.equals("max"))
        return new Float(cEx.maxQ);
      if (colName.contains("min") && colName.contains("max")) {
        double values[]={cEx.minQ,cEx.maxQ,qMin,qMax};
        return values;
      }
      if (colName.contains("uses"))
        return new Integer(cEx.nUses);
      if (colName.equals("n+") || colName.equals("n +"))
        return new Integer(cEx.nCasesRight);
      if (colName.equals("n-") || colName.equals("n -"))
        return new Integer(cEx.nCasesWrong);
      if (colName.contains("+/-"))
        if (cEx.nCasesRight+cEx.nCasesWrong>0)
          return new Float(cEx.nCasesRight*1f/(cEx.nCasesRight+cEx.nCasesWrong));
        else
          return null;

      if (colName.equals("x") || colName.equals("x1d")) //in 1D projection
        return new Float(cEx.x1D);

      if (colName.equals("order"))
        return (order == null) ? new Integer(row) : new Integer(order[row]);
      if (colName.contains("cluster"))
        return (clusters == null) ? new Integer(-1) : new Integer(clusters[row]);
      if (colName.contains("conditions"))
        return new Integer(cEx.eItems.length);
      if (colName.equalsIgnoreCase(columnNamesUnited[0]))
        return (cEx instanceof UnitedRule) ? ((UnitedRule) cEx).nOrigRight : 1;
      if (colName.equalsIgnoreCase(columnNamesUnited[1]))
        return (cEx instanceof UnitedRule) ? ((UnitedRule) cEx).nOrigWrong : 0;
      if (colName.equalsIgnoreCase(columnNamesUnited[2]))
        if (cEx instanceof UnitedRule) {
          UnitedRule r = (UnitedRule) cEx;
          return 100f * r.nOrigRight / (r.nOrigRight + r.nOrigWrong);
        }
        else
          return 100f;
      if (colName.equalsIgnoreCase(columnNamesUnited[3]))
        return (cEx instanceof UnitedRule) ? ((UnitedRule) cEx).countFromRules() : 0;
      if (colName.equalsIgnoreCase(columnNamesUnited[4]))
        return (cEx instanceof UnitedRule) ? ((UnitedRule) cEx).getHierarchyDepth() : 0;
    }
    return null;
  }

  public void putTableToClipboard () {
    String s="";
    boolean first=true;
    for (int c=0; c<getColumnCount(); c++) {
      Object v=getValueAt(0,c);
      if (v instanceof Integer || v instanceof Float || v instanceof String || v instanceof Double) {
        s += ((first) ? "" : ",") + getColumnName(c);
        first=false;
      }
    }
    s+="\n";
    for (int r=0; r<getRowCount(); r++) {
      first=true;
      for (int c=0; c<getColumnCount(); c++) {
        Object v=getValueAt(r,c);
        if (v instanceof Integer || v instanceof Float || v instanceof String || v instanceof Double) {
          s += ((first) ? "" : ",") + v;
          first=false;
        }
      }
      s+="\n";
    }
    StringSelection stringSelection = new StringSelection(s);
    Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
    clipboard.setContents(stringSelection, null);
  }

  public float getColumnMax(int col) {
    float max=Float.NaN;
    for (int i=0; i<getRowCount(); i++) {
      Object v=getValueAt(i,col);
      if (v==null)
        continue;
      if (v instanceof Integer) {
        Integer iv=(Integer)v;
        if (Double.isNaN(max) || max<iv)
          max=iv;
      }
      else
      if (v instanceof Float) {
        Float fv=(Float)v;
        if (Double.isNaN(max) || max<fv)
          max=fv;
      }
      else
      if (v instanceof Double) {
        Double dv=(Double)v;
        if (Double.isNaN(max) || max<dv)
          max=dv.floatValue();
      }
      else
      if (v instanceof double[]) {
        double d[]=(double[])v, dv=d[Math.min(1,d.length-1)];
        if (Double.isNaN(max) || max<dv)
          max=(float)dv;
      }
      else
        return Float.NaN;
  }
    return max;
  }

  public float getColumnMin(int col) {
    float min=Float.NaN;
    for (int i=0; i<getRowCount(); i++) {
      Object v=getValueAt(i,col);
      if (v==null)
        continue;
      if (v instanceof Integer) {
        Integer iv=(Integer)v;
        if (Double.isNaN(min) || min>iv)
          min=iv;
      }
      else
      if (v instanceof Float) {
        Float fv=(Float)v;
        if (Double.isNaN(min) || min>fv)
          min=fv;
      }
      else
      if (v instanceof Double) {
        Double dv=(Double)v;
        if (Double.isNaN(min) || min>dv)
          min=dv.floatValue();
      }
      else
      if (v instanceof double[]) {
        double d[]=(double[])v, dv=d[0];//d[d.length-1];
        if (Double.isNaN(min) || min>dv)
          min=(float)dv;
      }
      else
        return Float.NaN;
    }
    return min;
  }

  public void stateChanged(ChangeEvent e) {
    if (e.getSource() instanceof MySammonsProjection) {
      MySammonsProjection sam=(MySammonsProjection)e.getSource();
      double proj[][]=(sam.done)?sam.getProjection():sam.bestProjection;
      if (proj==null)
        return;
      if (proj[0].length==1) { // 1D projection
        System.out.println("Table: update 1D projection coordinates (column X)");
        for (int i=0; i<proj.length && i<exList.size(); i++)
          exList.get(i).x1D=proj[i][0];
      }
      fireTableDataChanged();
    }
    else
      if (e.getSource() instanceof ClustersAssignments)
        setCusterAssignments((ClustersAssignments)e.getSource());
  }
}
