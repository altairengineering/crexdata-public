package TapasExplTreeViewer.ui;

import TapasExplTreeViewer.rules.ClassConfusionMatrix;
import TapasExplTreeViewer.rules.DataSet;
import TapasExplTreeViewer.vis.MatrixPainter;

import javax.swing.*;
import java.awt.*;

public class DataVersionsViewer extends JFrame {
  public static int InstanceN=0;
  public DataSet origData=null;
  protected JTabbedPane dataTabbedPane=null;

  public DataVersionsViewer(DataTableViewer dViewer, ClassConfusionMatrix cMatrix, String rulesInfoText) {
    super();
    if (dViewer==null)
      return;
    this.origData=dViewer.getData();
    setTitle("Data from file "+origData.filePath);
    dataTabbedPane =new JTabbedPane();
    add(dataTabbedPane, BorderLayout.CENTER);
  
    JSplitPane splitPane=null;
    if (cMatrix==null)
      dataTabbedPane.addTab(origData.versionLabel+" (original data)",dViewer);
    else {
      JPanel mp=makeMatrixPanel(cMatrix,rulesInfoText);
      dViewer.setMinimumSize(new Dimension(200,200));
      mp.setMinimumSize(new Dimension(200,200));
      splitPane=new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, dViewer, mp);
      splitPane.setOneTouchExpandable(true); // Add little arrows to quickly expand/collapse
      splitPane.setContinuousLayout(true); // Enable continuous layout
      dataTabbedPane.addTab("version " + dViewer.getData().versionLabel, splitPane);
    }
    Dimension scSize=Toolkit.getDefaultToolkit().getScreenSize();
    setSize(Math.round(0.8f*scSize.width),Math.round(0.8f*scSize.height));
    ++InstanceN;
    setLocation(InstanceN*50,InstanceN*25);
    setVisible(true);
    if (splitPane!=null)
      splitPane.setDividerLocation(0.75);
  }

  public void addDataViewer(DataTableViewer dViewer, ClassConfusionMatrix cMatrix, String rulesInfoText) {
    JSplitPane splitPane=null;
    if (cMatrix == null)
      dataTabbedPane.addTab("version " + dViewer.getData().versionLabel, dViewer);
    else {
      JPanel mp=makeMatrixPanel(cMatrix,rulesInfoText);
      dViewer.setMinimumSize(new Dimension(200,200));
      mp.setMinimumSize(new Dimension(200,200));
      splitPane=new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, dViewer, mp);
      splitPane.setOneTouchExpandable(true); // Add little arrows to quickly expand/collapse
      splitPane.setContinuousLayout(true); // Enable continuous layout
      splitPane.setResizeWeight(0.7); // Optional: balance initial position
      dataTabbedPane.addTab("version " + dViewer.getData().versionLabel, splitPane);
    }
    dataTabbedPane.setSelectedIndex(dataTabbedPane.getComponentCount()-1);
    if (splitPane!=null)
      splitPane.setDividerLocation(0.75);
    toFront();
  }

  public JPanel makeMatrixPanel(ClassConfusionMatrix cMatrix, String rulesInfoText) {
    if (cMatrix==null || cMatrix.classNs==null || cMatrix.counts==null || cMatrix.nDataTotal<1)
      return null;

    String labels[]=new String[cMatrix.classNs.size()];
    for (int i=0; i<cMatrix.classNs.size(); i++)
      labels[i]=cMatrix.classNs.get(i).toString();

    // Create the tabbed pane
    JTabbedPane tabbedPane = new JTabbedPane();

    // Create the MatrixPainter instance for the count matrix
    MatrixPainter countMatrixPainter = new MatrixPainter(cMatrix.counts);
    countMatrixPainter.setShowValues(true);
    countMatrixPainter.setShowColumnTotals(true);
    countMatrixPainter.setMinValue(0);
    countMatrixPainter.setLabels(labels);
    JScrollPane countScrollPane = new JScrollPane(countMatrixPainter);
    tabbedPane.addTab("Counts", countScrollPane);

    // Create the MatrixPainter instance for the percentage matrix
    MatrixPainter percentageMatrixPainter = new MatrixPainter(cMatrix.percents,false);
    percentageMatrixPainter.setShowValues(true);
    percentageMatrixPainter.setShowColumnTotals(false);
    percentageMatrixPainter.setMinValue(0);
    percentageMatrixPainter.setLabels(labels);
    JScrollPane percentageScrollPane = new JScrollPane(percentageMatrixPainter);
    tabbedPane.addTab("Percentages", percentageScrollPane);

    JTextArea infoArea=new JTextArea("Rule set: "+rulesInfoText);
    infoArea.setLineWrap(true);
    infoArea.setWrapStyleWord(true);
    infoArea.append("\nResults of applying the rules to "+cMatrix.nDataTotal+" data records. ");
    if (cMatrix.data!=null && cMatrix.data.description!=null) {
      infoArea.append("\nDataVersion: "+cMatrix.data.description);
      if (cMatrix.data.previousVersion!=null && cMatrix.data.previousVersion.description!=null)
        infoArea.append("\nPrevious version of the data: "+cMatrix.data.previousVersion.description+"\n");
    }
    double accuracy=100.0*cMatrix.nSame/cMatrix.nDataTotal;
    infoArea.append("Number of classes: "+cMatrix.counts.length+
        "; number of correct class assignments: "+cMatrix.nSame+
        String.format(" (%.2f %%)",accuracy)+
        "; number of wrong class assignments: "+(cMatrix.nDataTotal-cMatrix.nSame)+
        String.format(" (%.2f %%)",100.0-accuracy));

    JSplitPane splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, tabbedPane, infoArea);
    splitPane.setResizeWeight(0.8); // Optional: balance initial position
    add(splitPane,BorderLayout.CENTER);

    JPanel p=new JPanel();
    p.setLayout(new BorderLayout());
    p.add(splitPane,BorderLayout.CENTER);

    return p;
  }
}
