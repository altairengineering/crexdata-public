package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;
import TapasExplTreeViewer.clustering.ReachabilityPlot;
import TapasExplTreeViewer.rules.RuleSet;
import TapasExplTreeViewer.vis.ExplanationsProjPlot2D;
import TapasExplTreeViewer.vis.ProjectionPlot2D;
import TapasUtilities.ItemSelectionManager;
import TapasUtilities.RenderLabelBarChart;
import TapasUtilities.SingleHighlightManager;
import TapasUtilities.TableRowsSelectionManager;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.LineBorder;
import javax.swing.table.TableCellRenderer;
import javax.swing.table.TableColumn;
import javax.swing.table.TableRowSorter;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Vector;

public class RulesTableViewer extends JPanel implements RulesOrderer {
  public static Border highlightBorder=new LineBorder(ProjectionPlot2D.highlightColor,1);

  public RuleSet ruleSet=null;

  public SingleHighlightManager highlighter=null;
  public ItemSelectionManager selector=null;

  protected ExListTableModel tblModel=null;
  protected JTable table=null;
  protected JLabel_Rule ruleRenderer=null;
  protected JTextArea infoArea=null;

  /**
   * used for rendering rules in tooltips
   */
  protected Vector<String> attrs=null;
  protected Vector<float[]> minMax =null;
  /**
   * Additional frames created by the instance of ShowRules that owns the RulesTableViewer.
   * These frames are only relevant to this instance
   * and are therefore closed when the instance is closed.
   */
  protected ArrayList<JFrame> frames=null;

  public RulesTableViewer (RuleSet ruleSet, SingleHighlightManager highlighter, ItemSelectionManager selector) {
    super();
    this.ruleSet=ruleSet; this.highlighter=highlighter; this.selector=selector;

    makeTable();

    JScrollPane scrollPane = new JScrollPane(table);
    infoArea = new JTextArea(ruleSet.description);
    infoArea.setLineWrap(true);
    infoArea.setWrapStyleWord(true);

    JSplitPane splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, scrollPane, infoArea);
    splitPane.setResizeWeight(0.95); // balance initial position

    setLayout(new BorderLayout());
    add(splitPane,BorderLayout.CENTER);
    Component owner=this;
    addComponentListener(new ComponentAdapter() {
      @Override
      public void componentShown(ComponentEvent e) {
        super.componentShown(e);
        splitPane.setDividerLocation(0.95);
        owner.removeComponentListener(this);
      }
    });
  }

  protected void makeTable () {
    tblModel=new ExListTableModel(ruleSet.rules,
        ruleSet.attrMinMax,ruleSet.orderedFeatureNames);
    ruleSet.listOfFeatures=tblModel.listOfFeatures;

    attrs=new Vector(tblModel.listOfFeatures.size());
    minMax =new Vector<>(tblModel.listOfFeatures.size());
    for (int i=0; i<tblModel.listOfFeatures.size(); i++) {
      String s=tblModel.listOfFeatures.get(i);
      attrs.add(s);
      minMax.add(ruleSet.attrMinMax.get(s));
    }

    table=new JTable(tblModel) {
      public String getToolTipText(MouseEvent e) {
        java.awt.Point p = e.getPoint();
        int rowIndex = rowAtPoint(p);
        if (rowIndex>=0) {
          int realRowIndex = convertRowIndexToModel(rowIndex);
          highlighter.highlight(new Integer(realRowIndex));
          int colIndex=columnAtPoint(p);
          String s="";
          if (colIndex>=0) {
            int realColIndex=convertColumnIndexToModel(colIndex);
            s=tblModel.getColumnName(realColIndex);
          }
          CommonExplanation ce=ruleSet.getRule(realRowIndex);
          Vector<CommonExplanation> vce=null;
          ArrayList selected=selector.getSelected();
          if (selected!=null && selected.size()>0) {
            vce=new Vector<>(selected.size());
            for (int i = 0; i < selected.size(); i++)
              vce.add(ruleSet.getRule((Integer)selected.get(i)));
          }
          try {
            BufferedImage bi = ShowSingleRule.getImageForRule(300,100, ce, vce, attrs, minMax);
            File outputfile = new File("img.png");
            ImageIO.write(bi, "png", outputfile);
            //System.out.println("img"+ce.numId+".png");
          } catch (IOException ex) { System.out.println("* error while writing image to file: "+ex.toString()); }
          String out=ce.toHTML(ruleSet.listOfFeatures,ruleSet.attrMinMax,s,"img.png");
          //System.out.println(out);
          return out;
        }
        highlighter.clearHighlighting();
        return "";
      }

      public Component prepareRenderer(TableCellRenderer renderer, int row, int column) {
        if (renderer==null)
          return null;
        Component c = super.prepareRenderer(renderer, row, column);
        Color bkColor=(isRowSelected(row))?getSelectionBackground():getBackground();
        int rowIdx=convertRowIndexToModel(row), colIdx=convertColumnIndexToModel(column);
        boolean isCluster=tblModel.isClusterColumn(colIdx);
        if (isCluster)
          bkColor= ReachabilityPlot.getColorForCluster((Integer)tblModel.getValueAt(rowIdx,colIdx));
        boolean isAction=false, isQ=false;
        RuleSet origRS=ruleSet.getOriginalRuleSet();
        if (!isCluster && origRS.minAction<origRS.maxAction &&
            tblModel.isResultClassColumn(colIdx)) {
          bkColor = ExplanationsProjPlot2D.getColorForAction((Integer) tblModel.getValueAt(rowIdx, colIdx),
              origRS.minAction, origRS.maxAction);
          isAction=true;
        }
        if (!isCluster && origRS.minQValue<origRS.maxQValue && tblModel.isResultValueColumn(colIdx)) {
          Object v=tblModel.getValueAt(rowIdx, colIdx);
          if (v!=null)
            if (v instanceof Float) {
              bkColor = ExplanationsProjPlot2D.getColorForQ(new Double((Float)v),origRS.minQValue, origRS.maxQValue);
              isQ = true;
            }
            else
            if (v instanceof double[]) {
              double d[]=(double[])v;
              if (d.length>1) {
                Color qC=ExplanationsProjPlot2D.getColorForQ((d[0]+d[1])/2,origRS.minQValue, origRS.maxQValue);
                bkColor = new Color(qC.getRed(),qC.getGreen(),qC.getBlue(),30);
                isQ = true;
              }
            }
        }
        c.setBackground(bkColor);
        if (highlighter==null || highlighter.getHighlighted()==null ||
            ((Integer)highlighter.getHighlighted())!=rowIdx) {
          ((JComponent) c).setBorder(null);
          return c;
        }
        ((JComponent) c).setBorder(highlightBorder);
        if (!isCluster && !isAction && !isQ)
          c.setBackground(ProjectionPlot2D.highlightFillColor);
        return c;
      }
    };

    table.addMouseListener(new MouseAdapter() {
      private void reactToMousePosition(MouseEvent e) {
        int rowIndex=table.rowAtPoint(e.getPoint());
        if (rowIndex<0)
          highlighter.clearHighlighting();
        else {
          int realRowIndex = table.convertRowIndexToModel(rowIndex);
          highlighter.highlight(new Integer(realRowIndex));
        }
      }
      @Override
      public void mouseEntered(MouseEvent e) {
        reactToMousePosition(e);
        super.mouseEntered(e);
      }

      @Override
      public void mouseExited(MouseEvent e) {
        highlighter.clearHighlighting();
        super.mouseExited(e);
      }

      @Override
      public void mouseMoved(MouseEvent e) {
        reactToMousePosition(e);
        super.mouseMoved(e);
      }
    });

    for (int i=0; i<tblModel.getColumnCount(); i++) {
      if (tblModel.isMinMaxColumn(i)) {
        JLabel_Subinterval subIntRend = new JLabel_Subinterval();
        subIntRend.setDrawTexts(true);
        subIntRend.setPrecision(3);
        table.getColumnModel().getColumn(i).setCellRenderer(subIntRend);
      }
      else
      if (!tblModel.isClusterColumn(i) && !tblModel.isResultClassColumn(i)) {
        Class columnClass = tblModel.getColumnClass(i);
        if (columnClass==null)
          continue;
        if (columnClass.equals(Integer.class) ||
            columnClass.equals(Float.class) ||
            columnClass.equals(Double.class)) {
          float min = tblModel.getColumnMin(i), max = tblModel.getColumnMax(i);
          RenderLabelBarChart rBar = new RenderLabelBarChart(min, max);
          table.getColumnModel().getColumn(i).setCellRenderer(rBar);
        }
      }
    }

    ruleRenderer=new JLabel_Rule();
    ruleRenderer.setAttrs(attrs, minMax);
    int rIdx=tblModel.getRuleColumnIdx();
    if (rIdx>=0) {
      table.getColumnModel().getColumn(rIdx).setCellRenderer(ruleRenderer);
      table.getColumnModel().getColumn(rIdx).setPreferredWidth(200);
    }

    for (int i=0; i<tblModel.listOfFeatures.size(); i++)
      table.getColumnModel().getColumn(tblModel.listOfColumnNames.size()+i).
          setCellRenderer(new JLabel_Subinterval());

    /**/
    TableRowsSelectionManager rowSelMan=new TableRowsSelectionManager();
    rowSelMan.setTable(table);
    rowSelMan.setHighlighter(highlighter);
    rowSelMan.setSelector(selector);
    rowSelMan.setMayScrollTable(false);
    /**/

    Dimension size=Toolkit.getDefaultToolkit().getScreenSize();

    table.setPreferredScrollableViewportSize(new Dimension(Math.round(size.width * 0.7f), Math.round(size.height * 0.8f)));
    table.setFillsViewportHeight(true);
    table.setAutoCreateRowSorter(true);
    table.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
    table.setRowSelectionAllowed(true);
    table.setColumnSelectionAllowed(false);

    TableRowSorter sorter=(TableRowSorter)table.getRowSorter();
    for (int i=0; i<tblModel.listOfFeatures.size(); i++)
      sorter.setComparator(tblModel.listOfColumnNames.size()+i, new Comparator<Object>() {
        public int compare (Object o1, Object o2) {
          if (o1 instanceof double[]) {
            double d1=((double[])o1)[0], d2=((double[])o2)[0], d1r=((double[])o1)[1], d2r=((double[])o2)[1];
            if (Double.isNaN(d1) && Double.isNaN(d2))
              return 0;
            if (Double.isNaN(d1))
              return 1;
            if (Double.isNaN(d2))
              return -1;
            return (d1<d2)?-1:(d1==d2)?((d1r<d2r)?-1:(d1r==d2r)?0:1):1;
          }
          else
            return 0;
        }
      });
  }

  public void updateDataInTable () {
    if (table==null || tblModel==null)
      return;
    for (int i=0; i<tblModel.getColumnCount(); i++)
      if (!tblModel.isClusterColumn(i) && !tblModel.isResultClassColumn(i)) {
        TableColumn cMod=table.getColumnModel().getColumn(i);
        if (cMod.getCellRenderer() instanceof RenderLabelBarChart) {
          float min = tblModel.getColumnMin(i), max = tblModel.getColumnMax(i);
          RenderLabelBarChart rBar = (RenderLabelBarChart)cMod.getCellRenderer();
          rBar.setMinMax(min,max);
        }
      }
    tblModel.fireTableDataChanged();
  }

  public JTable getTable() {
    return table;
  }

  public String getGeneralInfo () {
    return infoArea.getText();
  }

  public Vector<String> getAttrs() {
    return attrs;
  }

  public Vector<float[]> getMinMax() {
    return minMax;
  }

  public ExListTableModel getTableModel() {
    return tblModel;
  }

  public JLabel_Rule getRuleRenderer() {
    return ruleRenderer;
  }

  public int[] getRulesOrder(ArrayList rules) {
    if (rules==null || rules.isEmpty() || ruleSet.rules==null || table==null)
      return null;
    int order[]=new int[rules.size()];
    int k=0;
    for (int i=0; i<table.getRowCount(); i++) {
      int mIdx = table.convertRowIndexToModel(i);
      int idx=rules.indexOf(ruleSet.rules.get(mIdx));
      if (idx>=0)
        order[k++]=idx;
    }
    return order;
  }

  public void addFrame(JFrame frame) {
    if (frame==null)
      return;
    if (frames==null)
      frames=new ArrayList<JFrame>(20);
    frames.add(frame);
    frame.addWindowListener(new WindowAdapter() {
      @Override
      public void windowClosing(WindowEvent e) {
        super.windowClosing(e);
        if (frames!=null)
          frames.remove(frame);
      }
    });
    frame.toFront();
  }

  public void closeAllFrames() {
    if (frames==null)
      return;
    for (JFrame frame:frames)
      frame.dispose();
    frames.clear();
  }
}
