package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;
import TapasExplTreeViewer.clustering.ClusterContent;
import TapasUtilities.RenderLabelBarChart;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableColumnModel;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Vector;

public class ClustersTable extends JPanel {

  protected ClusterContent clusters[]=null;
  protected double distanceMatrix[][];
  public JScrollPane scrollPane=null;

  public ClustersTable (ClusterContent clusters[], double distanceMatrix[][], ArrayList<CommonExplanation> exList,
                        JLabel_Rule ruleRenderer, Hashtable<String,float[]> attrMinMax, int minA, int maxA, double minQ, double maxQ) {
    super();
    this.clusters=clusters;
    this.distanceMatrix=distanceMatrix;
    JTable table=new JTable(new ClustersTableModel(clusters,distanceMatrix,exList,minA,maxA,minQ,maxQ)){
      public String getToolTipText(MouseEvent e) {
        java.awt.Point p = e.getPoint();
        int rowIndex = rowAtPoint(p);
        if (rowIndex>=0) {
          int realRowIndex = convertRowIndexToModel(rowIndex);
          String s="";
          CommonExplanation ce=(CommonExplanation)exList.get(clusters[realRowIndex].medoidIdx);
          Vector<CommonExplanation> vce=null;
          vce=new Vector<>();
          for (int i=0; i<clusters[realRowIndex].member.length; i++)
            if (clusters[realRowIndex].member[i]) {
              //int idx=clusters[realRowIndex].objIds[i]; // objIds==null
              CommonExplanation ce1=(CommonExplanation)exList.get(i);
              vce.add(ce1);
            }
          try {
            BufferedImage bi = ShowSingleRule.getImageForRule(300,100, ce, vce, ruleRenderer.attrs, ruleRenderer.minmax);
            File outputfile = new File("img.png");
            ImageIO.write(bi, "png", outputfile);
          } catch (IOException ex) { System.out.println("* error while writing image to file: "+ex.toString()); }
          String out=ce.toHTML(null,attrMinMax,s,"img.png");
          return out;
        }
        return "";
      }

    };
    table.setFillsViewportHeight(true);
    table.setAutoCreateRowSorter(true);
    table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
    table.setRowSelectionAllowed(true);
    table.setColumnSelectionAllowed(false);
    table.setRowHeight((int)(1.5*table.getRowHeight()));
    TableColumnModel tableColumnModel=table.getColumnModel();
    tableColumnModel.getColumn(0).setCellRenderer(new RenderLabelBarChart(0,clusters.length-1));
    int maxSize=clusters[0].getMemberCount(), maxUses=0;
    float maxD=0;
    for (int i=0; i<clusters.length; i++) {
      maxSize = Math.max(maxSize, clusters[i].getMemberCount());
      maxD = Math.max(maxD,(float)clusters[i].getDiameter(distanceMatrix));
      maxUses = Math.max(maxUses, (int)table.getModel().getValueAt(i,2));
    }
    tableColumnModel.getColumn(1).setCellRenderer(new RenderLabelBarChart(0,maxSize));
    tableColumnModel.getColumn(2).setCellRenderer(new RenderLabelBarChart(0,maxUses));
    for (int i=3; i<=4; i++)
      tableColumnModel.getColumn(i).setCellRenderer(new RenderLabelBarChart(0,maxD));
    tableColumnModel.getColumn(5).setCellRenderer(ruleRenderer);
    for (int i=6; i<=7; i++)
      tableColumnModel.getColumn(i).setCellRenderer(new JLabel_Bars());
    for (int i=0; i<5; i++)
      tableColumnModel.getColumn(i).setPreferredWidth((i<3)?30:50);

    scrollPane = new JScrollPane(table);
    scrollPane.setMinimumSize(new Dimension(100,200));
    scrollPane.setOpaque(true);
  }
}

class ClustersTableModel extends AbstractTableModel {

  protected ClusterContent clusters[]=null;
  protected double distanceMatrix[][]=null;
  protected ArrayList<CommonExplanation> exList=null;
  protected int minA, maxA;
  protected double minQ, maxQ;

  public ClustersTableModel (ClusterContent clusters[], double distanceMatrix[][], ArrayList<CommonExplanation> exList,
                             int minA, int maxA, double minQ, double maxQ) {
    this.clusters=clusters;
    this.distanceMatrix=distanceMatrix;
    this.exList=exList;
    this.minA=minA; this.maxA=maxA;
    this.minQ=minQ; this.maxQ=maxQ;
  }

  private String columnNames[] = {"Cluster","Size","N uses","m-Radius","Diameter","Rule","Action","Q"};
  public int getColumnCount() {
    return columnNames.length;
  }
  public String getColumnName(int col) {
    return columnNames[col];
  }
  public int getRowCount() {
    return clusters.length;
  }
  public Class getColumnClass(int c) {
    return (getValueAt(0, c)==null) ? null: getValueAt(0, c).getClass();
  }
  public Object getValueAt (int row, int col) {
    switch (col) {
      case 0:
        return row;
      case 1:
        return clusters[row].getMemberCount();
      case 2:
        int n=0;
        for (int i=0; i<clusters[row].member.length; i++)
          if (clusters[row].member[i])
            n+=((CommonExplanation)exList.get(i)).nUses;
        return n;
      case 3:
        return clusters[row].getMRadius(distanceMatrix);
      case 4:
        return clusters[row].getDiameter(distanceMatrix);
      case 5:
        return exList.get(clusters[row].medoidIdx);
      case 6:
        int len=maxA-minA+1;
        int counts[]=new int[len];
        for (int i=0; i<len; i++)
          counts[i]=0;
        if (len>1)
          for (int i=0; i<clusters[row].member.length; i++)
            if (clusters[row].member[i])
              counts[((CommonExplanation)exList.get(i)).action-minA]++;
        return counts;
      case 7:
        counts=null;
        if (maxQ==minQ) {
          counts=new int[1];
          counts[0]=0;
        }
        else {
          len = 10;
          counts = new int[len];
          for (int i = 0; i < len; i++)
            counts[i] = 0;
          for (int i=0; i<clusters[row].member.length; i++)
            if (clusters[row].member[i])
              counts[(int)Math.floor((len-1)*(((CommonExplanation)exList.get(i)).meanQ-minQ)/(maxQ-minQ))]++;
        }
        return counts;
    }
    return 0;
  }
}