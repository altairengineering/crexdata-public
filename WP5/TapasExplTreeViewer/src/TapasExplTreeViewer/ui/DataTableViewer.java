package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;
import TapasExplTreeViewer.rules.DataRecord;
import TapasExplTreeViewer.rules.DataSet;
import TapasExplTreeViewer.rules.RuleMaster;
import TapasExplTreeViewer.util.CSVDataLoader;
import TapasUtilities.ItemSelectionManager;
import TapasUtilities.TableRowsSelectionManager;

import javax.swing.*;
import javax.swing.table.TableRowSorter;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;

public class DataTableViewer extends JPanel {
  private DataSet data=null;
  private RulesPresenter rulesPresenter=null;
  private JTable dataTable=null;
  private DataTableModel tableModel=null;
  private JTextArea infoArea=null;
  private int shownRow=-1;
  private ItemSelectionManager selector=null;
  private JSplitPane splitPane=null;
  
  public DataTableViewer (DataSet dataSet,
                          String featureNames[],
                          RulesPresenter rulesPresenter) {
    if (dataSet==null || dataSet.records==null || dataSet.records.isEmpty())
      return;
    data=dataSet; this.rulesPresenter=rulesPresenter;
    tableModel=new DataTableModel(dataSet,featureNames);
    dataTable = new JTable(tableModel){
      public String getToolTipText(MouseEvent e) {
        int row = dataTable.rowAtPoint(e.getPoint());
        if (row<0)
          return null;
        int trueRow=convertRowIndexToModel(row);
        //System.out.println("Row at pointer: "+row+", true row = "+trueRow);
        return getPopupContent(trueRow);
      }
    };
    
    for (int i = 1; i < tableModel.getColumnCount(); i++)
      if (tableModel.isNumericColumn(i)) {
        double minmax[]=tableModel.getColumnMinMax(i);
        if (minmax!=null)
          dataTable.getColumnModel().getColumn(i).setCellRenderer(new NumericCellRenderer(minmax[0], minmax[1]));
      }
  
    TableRowSorter<DataTableModel> sorter = new TableRowSorter<DataTableModel>(tableModel);
    dataTable.setRowSorter(sorter);
    dataTable.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
    dataTable.setRowSelectionAllowed(true);
    dataTable.setColumnSelectionAllowed(false);

    selector=new ItemSelectionManager();
    TableRowsSelectionManager rowSelMan=new TableRowsSelectionManager();
    rowSelMan.setTable(dataTable);
    rowSelMan.setSelector(selector);
    rowSelMan.setMayScrollTable(false);

    JPopupMenu menu=new JPopupMenu();
    JMenuItem mit;
    if (rulesPresenter!=null) {
      menu.add(mit=new JMenuItem("Extract the rules applicable to the selected data"));
      mit.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          showRulesApplicableToData();
        }
      });
    }

    menu.add(mit=new JMenuItem("Export the data to a CSV file"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        exportDataToCSVFile();
      }
    });
    
    dataTable.addMouseListener(new MouseAdapter() {
      @Override
      public void mousePressed(MouseEvent e) {
        if (SwingUtilities.isRightMouseButton(e)) {
          menu.show(e.getComponent(), e.getX(), e.getY());
        }
        super.mousePressed(e);
      }
    });

    JScrollPane scrollPane = new JScrollPane(dataTable);
    setLayout(new BorderLayout());

    if (data.description!=null) {
      String info=data.description;
      if (data.previousVersion!=null)
        info+="\nPrevious version of the data: "+data.previousVersion.description;
      infoArea=new JTextArea(info);
      infoArea.setLineWrap(true);
      infoArea.setWrapStyleWord(true);
      splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, scrollPane, infoArea);
      splitPane.setDividerLocation(0.8);
      splitPane.setResizeWeight(0.8); // Optional: balance initial position
      add(splitPane,BorderLayout.CENTER);
    }
    else
      add(scrollPane, BorderLayout.CENTER);

    if (splitPane!=null) {
      Component owner=this;
      addComponentListener(new ComponentAdapter() {
        @Override
        public void componentShown(ComponentEvent e) {
          super.componentShown(e);
          splitPane.setDividerLocation(0.9);
          owner.removeComponentListener(this);
        }
      });
    }
  }

  public DataSet getData() {
    return data;
  }

  public String getPopupContent(int row) {
    if (row<0)
      return null;
    // Get the record ID and row number
    String recordId = tableModel.getValueAt(row, 0).toString();
    String rowNumber = String.valueOf(row + 1);
  
    // Build the HTML content
    StringBuilder htmlContent = new StringBuilder("<html><body style='background-color:#FFFACD; padding: 5px;'>");
    htmlContent.append("<div style='text-align: center;'>")
        .append("<b>Record ID:</b> ").append(recordId)
        .append("<br><b>Row:</b> ").append(rowNumber)
        .append("</div>");
    htmlContent.append("<br><table border='1' cellspacing='0' cellpadding='2'>");
    htmlContent.append("<tr><th>Column</th><th>Value</th></tr>");
  
    for (int col = 0; col < tableModel.getColumnCount(); col++) {
      String columnName = tableModel.getColumnName(col);
      if (columnName.equalsIgnoreCase("Record id"))
        continue;
      String value = tableModel.getValueAt(row, col) != null ? tableModel.getValueAt(row, col).toString() : "";
      htmlContent.append("<tr><td>").append(columnName).append("</td><td>").append(value).append("</td></tr>");
    }
  
    htmlContent.append("</table></body></html>");
    return htmlContent.toString();
  }
  
  public void exportDataToCSVFile() {
    if (data==null || data.records==null || data.records.isEmpty()) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "No data have been loaded in the system!","No data",
          JOptionPane.INFORMATION_MESSAGE);
      return;
    }
    String pathName=CSVDataLoader.selectFilePathThroughDialog(false);
    if (pathName==null)
      return;
    if (data.exportToCSV(pathName))
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Successfully exported the data to file "+pathName,"Data exported",
          JOptionPane.INFORMATION_MESSAGE);
    else
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Failed to export the data!","Export failed",
          JOptionPane.ERROR_MESSAGE);
  }
  
  public void showRulesApplicableToData(){
    if (rulesPresenter==null)
      return;
    ArrayList selected=selector.getSelected();
    if (selected==null || selected.isEmpty())
      return;
    ArrayList<CommonExplanation> rules=rulesPresenter.getRules();
    if (rules==null)
      return;
    ArrayList<DataRecord> records=new ArrayList<DataRecord>(selected.size());
    for (int i=0; i<selected.size(); i++) {
      int idx=(Integer)selected.get(i);
      records.add(data.records.get(idx));
    }
    ArrayList<CommonExplanation> applicableRules=RuleMaster.selectRulesApplicableToData(rules,records);
    if (applicableRules==null || applicableRules.isEmpty()) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "No rules applicable to the selected data have been found!","No applicable rules!",
          JOptionPane.ERROR_MESSAGE);
      return;
    }
    rulesPresenter.showRules(applicableRules,applicableRules.size()+" rules applicable to "+
                                                 records.size()+" selected data record(s)");
  }
}
