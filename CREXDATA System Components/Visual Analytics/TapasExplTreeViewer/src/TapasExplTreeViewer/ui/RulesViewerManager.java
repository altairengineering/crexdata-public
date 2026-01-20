package TapasExplTreeViewer.ui;

import TapasExplTreeViewer.rules.DataSet;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.util.ArrayList;

public class RulesViewerManager {
  public JFrame mainFrame=null;
  public JTabbedPane tabPane=null;
  public ArrayList<RulesTableViewer> viewers=null;
  /**
   * Frames that can be accessed by all ShowRules instances, e.g., frames showing data versions
   */
  protected ArrayList<JFrame> sharedFrames=null;
  /**
   * Register of temporary files created during the system's work,
   * to be deleted when the system is stopping the work.
   */
  protected ArrayList<File> createdFiles=new ArrayList<File>(20);

  /**
   * Data used for testing the rules
   */
  public DataSet currentData=null;
  /**
   * When a new dataset is loaded, the previously loaded ones remain accessible.
   * This is a list of all loaded datasets.
   */
  public ArrayList<DataSet> loadedData=null;
  public String dataFolder="";

  public int getRulesViewerCount() {
    if (viewers==null)
      return 0;
    return viewers.size();
  }

  public void addRulesViewer(RulesTableViewer rView) {
    if (rView==null)
      return;
    if (viewers==null)
      viewers=new ArrayList<RulesTableViewer>(30);
    viewers.add(rView);

    if (tabPane==null)
      tabPane=new JTabbedPane();
    tabPane.addTab(rView.ruleSet.versionLabel,rView);
    if (mainFrame==null) {
      mainFrame=new JFrame("Rules explorer v.30.01.2025 17:20");
      mainFrame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
      mainFrame.getContentPane().add(tabPane, BorderLayout.CENTER);
      Dimension d=Toolkit.getDefaultToolkit().getScreenSize();
      mainFrame.setSize(Math.round(0.8f*d.width), Math.round(0.8f*d.height));
      mainFrame.setLocation(Math.round(0.05f*d.width), Math.round(0.05f*d.height));
      mainFrame.setVisible(true);
      mainFrame.addWindowListener(new WindowAdapter() {
        @Override
        public void windowClosing(WindowEvent e) {
          if (mayQuit()) {
            super.windowClosing(e);
            mainFrame.dispose();
            eraseCreatedFiles();
            System.exit(0);
          }
        }
      });
    }
    else
      tabPane.setSelectedComponent(rView);
  }

  public int getViewerIndex(RulesTableViewer rView) {
    if (viewers==null)
      return -1;
    return viewers.indexOf(rView);
  }

  public void removeRulesViewer(RulesTableViewer rView) {
    int idx=getViewerIndex(rView);
    if (idx<0)
      return;
    viewers.remove(idx);
    tabPane.remove(rView);
    tabPane.revalidate();
    tabPane.repaint();
  }

  public void addLoadedDataSet(DataSet data) {
    if (data==null)
      return;
    if (loadedData==null)
      loadedData=new ArrayList<DataSet>(10);
    if (loadedData.size()>0)
      data.versionLabel=new Character((char)('A'+loadedData.size())).toString();
    loadedData.add(data);
  }

  public ArrayList<DataSet> getDataSetsList() {
    if (currentData==null)
      return loadedData;
    ArrayList<DataSet> dataSets=new ArrayList<DataSet>(20);
    dataSets.add(currentData);
    DataSet prev=currentData.previousVersion;
    while (prev!=null) {
      dataSets.add(prev);
      prev=prev.previousVersion;
    }
    DataSet.putAllVersionsInList(dataSets.get(dataSets.size()-1),dataSets);
    for (DataSet ds:loadedData)
      if (!dataSets.contains(ds))
        DataSet.putAllVersionsInList(ds,dataSets);
    return dataSets;
  }

  public void addSharedFrame(JFrame frame) {
    if (frame==null)
      return;
    if (sharedFrames==null)
      sharedFrames=new ArrayList<JFrame>(20);
    sharedFrames.add(frame);
  }

  public DataVersionsViewer findDataViewFrame(DataSet origData) {
    if (sharedFrames==null)
      return null;
    for (int i=0; i<sharedFrames.size(); i++)
      if (sharedFrames.get(i) instanceof DataVersionsViewer) {
        DataVersionsViewer dv=(DataVersionsViewer)sharedFrames.get(i);
        if (dv.origData.equals(origData))
          return dv;
      }
    return null;
  }

  public void eraseCreatedFiles () {
    if (createdFiles!=null && !createdFiles.isEmpty()) {
      for (File f : createdFiles)
        f.delete();
      createdFiles.clear();
    }
  }

  public boolean mayQuit() {
    int result = JOptionPane.showConfirmDialog(mainFrame,
        "Sure? Do you want to exit?",
        "Confirm quitting",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE);
    return result == JOptionPane.YES_OPTION;
  }
}
